# Assemble the final vessel set: TopBrain reference case (real annotation) + geometric/schematic vessels it lacks; enforce connections.
import numpy as np,pickle,json
from scipy.spatial import cKDTree
src=open('build_named.py').read();ns={}
exec(src.split("AC={")[0].split("A=json.load")[0],ns);exec("def smooth"+src.split("def smooth")[1].split("AC={")[0],ns)
tube,voxels,smooth=ns['tube'],ns['voxels'],ns['smooth']
N=pickle.load(open('out/named_vessels.pkl','rb'));S=pickle.load(open('out/schematic.pkl','rb'));TB=pickle.load(open('out/topbrain.pkl','rb'))
TB=[o for o in TB if o['id']!='tb_a3_l']                  # absent in the reference case (median A2/A3 supplies it): keep the real variant
# right P3-P4 is barely annotated in this case: mirror the left one
L=[o for o in TB if o['id']=='tb_p3p4_l'][0];R=[o for o in TB if o['id']=='tb_p3p4_r'][0]
v,f=L['mesh'];R['mesh']=(v*np.array([-1,1,1],np.float32),f[:,[0,2,1]]);R['P']=L['P']*np.array([-1,1,1]);R['src']='mirrored from the left (incompletely annotated in case 008)'
O=np.array([-90,-126,-72]);SH=(181,217,181);i,j,k=np.unravel_index(L['vox'],SH);x=-(i+O[0])-O[0]
R['vox']=np.ravel_multi_index((x.astype(int),j,k),SH).astype(np.uint32);R['radv']=L['radv']
DROP_N=lambda o:o['id'].startswith('a_') or o['id'] in ('vn_galen','vn_straight','vn_icv_l','vn_icv_r')
oldva={o['id']:o for o in N if o['id'] in('a_va_l','a_va_r')}
N=[o for o in N if not DROP_N(o)]
S=[o for o in S if not any(o['id'].startswith(p) for p in ('sa_sca2','sa_aica','sa_pica','sa_acha'))]
# extracranial vertebral artery (V2-V3) below the annotated V4
for s in 'lr':
    tb=[o for o in TB if o['id']=='tb_va_'+s][0];zmin=tb['P'][:,2].min()
    P=oldva['a_va_'+s]['P'];P=P[P[:,2]<zmin+4]
    r=np.full(len(P),1.8);mv,mf=tube(P,r);vox,rv=voxels(P,r)
    S.append(dict(id='sa_vaex_'+s,P=P,r=r,name='Vertebral artery (V2–V3, extracranial)',group='Arteries',color='#8a3324',side=s.upper(),vox=vox,radv=rv,mesh=(mv,mf),sch=True,note='schematic course: V2 in the transverse foramina, V3 over C1',kind='artery'))
for o in TB:o['sch']=False;o['note']=o['src']
D={o['id']:o for o in N+S+TB}
R=[]
for s in 'lr':
    R+=[(f'sa_heubner_{s}',0,[f'tb_a1a2_{s}']),(f'sa_cma_{s}',0,[f'tb_a3_{s}',f'tb_a1a2_{s}','tb_3rda3','tb_3rda2']),(f'sa_vaex_{s}',-1,[f'tb_va_{s}']),
        (f'sa_lsa1_{s}',0,[f'tb_m1_{s}']),(f'sa_lsa2_{s}',0,[f'tb_m1_{s}']),(f'sa_lsa3_{s}',0,[f'tb_m1_{s}']),
        (f'vn_bvr_{s}',-1,['tbv_vog']),(f'vn_ts_{s}',0,['vn_sss']),(f'vn_sig_{s}',0,[f'vn_ts_{s}']),
        (f'sv_labbe_{s}',-1,[f'vn_ts_{s}']),(f'sv_trolard_{s}',-1,['vn_sss']),(f'sv_trolard_{s}',0,[f'sv_smcv_{s}']),(f'sv_labbe_{s}',0,[f'sv_smcv_{s}']),
        (f'sv_tsv_{s}',-1,['tbv_icv']),(f'sv_septal_{s}',-1,['tbv_icv']),(f'sv_sps_{s}',-1,[f'vn_ts_{s}',f'vn_sig_{s}']),(f'sv_sps_{s}',0,[f'sv_cav_{s}']),
        (f'sv_dandy_{s}',-1,[f'sv_sps_{s}']),(f'sv_smcv_{s}',-1,[f'sv_cav_{s}'])]
    for jj in (1,2,3):R.append((f'sv_bridge{jj}_{s}',-1,['vn_sss']))
R+=[('vn_iss',-1,['tbv_sts','tbv_vog']),('sv_occ',0,['vn_sss'])]
def pts(o):return o['mesh'][0] if o.get('tb') else o['P']
def nearest(parents,q):
    best=None
    for pid in parents:
        if pid not in D:continue
        Q=pts(D[pid]);d=np.linalg.norm(Q-q,axis=1);i=int(d.argmin())
        if best is None or d[i]<best[0]:best=(d[i],pid,Q[i])
    return best
changed=set()
# vessels defined by their origin on a parent: regenerate from the parent's actual position instead of bending them
REGEN=('sa_heubner','sa_cma','sv_tsv','sv_septal')
for s in 'lr':
    m1=pts(D['tb_m1_'+s]);order=np.argsort(np.abs(m1[:,0]));m1=m1[order]
    for jj,frac in enumerate([.35,.6,.85]):
        q=m1[int(frac*(len(m1)-1))];sg=np.sign(q[0])
        cp=[q,q+[-sg*1,-1,6],q+[-sg*3,-3,16],q+[-sg*4,-4,24]]
        P=smooth(cp,s=0);r=np.full(len(P),0.45);o=D[f'sa_lsa{jj+1}_{s}'];o['P']=P;o['r']=r;changed.add(o['id'])
for cid,end,par in R:
    if cid.startswith(REGEN) and end==0 and cid in D:
        o=D[cid];P=o['P'];b=nearest(par,P[0])
        if b:
            gap,pid,tgt=b;cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))];keep=P[cum>=min(gap*0.6,12)][::4]
            o['P']=smooth(np.vstack([tgt,keep]),s=0);o['r']=np.full(len(o['P']),float(np.mean(o['r'])));changed.add(cid)
        continue
    if cid.startswith('sa_lsa'):continue

    if cid not in D:print('missing',cid);continue
    o=D[cid];P=o['P'].copy();q=P[end];b=nearest(par,q)
    if b is None:print('no parent',cid,par);continue
    gap,pid,tgt=b;seg=np.linalg.norm(np.diff(P,axis=0),axis=1);cum=np.r_[0,np.cumsum(seg)];dist=cum if end==0 else cum[-1]-cum
    w=np.clip(1-dist/max(6.0,gap*1.5),0,1)[:,None];o['P']=P+(tgt-q)[None,:]*w;changed.add(cid)
    if gap>2:print('snapped',cid,'->',pid,round(float(gap),1))
for c in changed:
    o=D[c];r=np.asarray(o['r']);mv,mf=tube(o['P'],r,seg=12 if c.startswith('vn_') else 10);vox,rv=voxels(o['P'],np.maximum(r,0.6));o['mesh']=(mv,mf);o['vox']=vox;o['radv']=rv
# connectors between two fixed meshes that do not touch
def connect(a,b):
    A,B=D[a]['mesh'][0],D[b]['mesh'][0];d,i=cKDTree(A).query(B);j=int(d.argmin());gap=float(d[j])
    if gap<0.8:return
    p0,p1=A[i[j]],B[j];P=np.linspace(p0,p1,max(3,int(gap/0.7)));r=np.full(len(P),0.9 if D[b]['group']=='Arteries' else 2.0)
    mv,mf=tube(P,r,seg=10);vox,rv=voxels(P,r);o=D[b];v,f=o['mesh'];o['mesh']=(np.vstack([v,mv]).astype(np.float32),np.vstack([f,mf+len(v)]).astype(np.int32))
    o['vox']=np.unique(np.r_[o['vox'],vox]).astype(np.uint32);o['radv']=np.r_[o['radv'],np.zeros(len(o['vox'])-len(o['radv']))][:len(o['vox'])]
    print('connector',a,'->',b,round(gap,1))
for a,b in [('tb_ba','tb_aica_l'),('tb_p1p2_r','tb_p3p4_r'),('vn_sss','tbv_sts')]:connect(a,b)
for o in TB:o['P']=pts(o)
pickle.dump([o for o in N]+TB,open('out/named_vessels.pkl','wb'));pickle.dump(S,open('out/schematic.pkl','wb'))
print('named',len(N)+len(TB),'schematic',len(S))
