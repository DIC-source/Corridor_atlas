# Enforce vessel connectivity: snap each branch's origin (and selected terminations) onto its parent's centerline.
import numpy as np,pickle
src=open('build_named.py').read();ns={}
exec(src.split("AC={")[0].split("A=json.load")[0],ns);exec("def smooth"+src.split("def smooth")[1].split("AC={")[0],ns)
tube,voxels=ns['tube'],ns['voxels']
N=pickle.load(open('out/named_vessels.pkl','rb'));S=pickle.load(open('out/schematic.pkl','rb'))
D={o['id']:o for o in N+S}
def side(p,s):return p.replace('{s}',s)
# (child, end, parent): end 0 = start of child, -1 = end of child; parent may be a list (nearest wins)
R=[]
for s in ['l','r']:
    R+=[(f'a_va_{s}',-1,['a_ba']),(f'a_p1p2_{s}',0,['a_ba']),(f'sa_sca2_{s}',0,['a_ba']),(f'a_m1_{s}',0,[f'a_ica_{s}']),(f'a_a1_{s}',0,[f'a_ica_{s}']),
        (f'a_pcom_{s}',0,[f'a_ica_{s}']),(f'a_pcom_{s}',-1,[f'a_p1p2_{s}']),(f'sa_acha_{s}',0,[f'a_ica_{s}']),(f'a_m2_{s}',0,[f'a_m1_{s}']),(f'a_p3_{s}',0,[f'a_p1p2_{s}']),
        (f'sa_heubner_{s}',0,[f'a_a1_{s}','a_a2']),(f'sa_cma_{s}',0,['a_a2']),(f'sa_aica_{s}',0,['a_ba']),(f'sa_pica_{s}',0,[f'a_va_{s}']),
        (f'sa_lsa1_{s}',0,[f'a_m1_{s}']),(f'sa_lsa2_{s}',0,[f'a_m1_{s}']),(f'sa_lsa3_{s}',0,[f'a_m1_{s}']),(f'a_a1_{s}',-1,['a_a2']),
        (f'vn_icv_{s}',-1,['vn_galen']),(f'vn_bvr_{s}',-1,['vn_galen']),(f'vn_ts_{s}',0,['vn_sss']),(f'vn_sig_{s}',0,[f'vn_ts_{s}']),
        (f'sv_labbe_{s}',-1,[f'vn_ts_{s}']),(f'sv_trolard_{s}',-1,['vn_sss']),(f'sv_trolard_{s}',0,[f'sv_smcv_{s}']),(f'sv_labbe_{s}',0,[f'sv_smcv_{s}']),
        (f'sv_tsv_{s}',-1,[f'vn_icv_{s}']),(f'sv_septal_{s}',-1,[f'vn_icv_{s}']),(f'sv_sps_{s}',-1,[f'vn_ts_{s}',f'vn_sig_{s}']),(f'sv_sps_{s}',0,[f'sv_cav_{s}']),
        (f'sv_dandy_{s}',-1,[f'sv_sps_{s}']),(f'sv_smcv_{s}',-1,[f'sv_cav_{s}'])]
    for j in (1,2,3):R.append((f'sv_bridge{j}_{s}',-1,['vn_sss']))
R+=[('vn_galen',-1,['vn_straight']),('vn_iss',-1,['vn_straight','vn_galen']),('vn_straight',-1,['vn_sss']),('sv_occ',0,['vn_sss'])]
def nearest(P,parents,q):
    best=None
    for pid in parents:
        if pid not in D:continue
        Q=D[pid]['P'];d=np.linalg.norm(Q-q,axis=1);i=int(d.argmin())
        if best is None or d[i]<best[0]:best=(d[i],pid,Q[i])
    return best
changed=set();rep=[]
for cid,end,par in R:
    if cid not in D:rep.append(('missing',cid));continue
    o=D[cid];P=o['P'].copy();q=P[end];b=nearest(P,par,q)
    if b is None:rep.append(('no parent',cid));continue
    gap,pid,tgt=b;rep.append((cid,'end' if end else 'start',pid,round(float(gap),1)))
    n=len(P);seg=np.linalg.norm(np.diff(P,axis=0),axis=1);cum=np.r_[0,np.cumsum(seg)]
    dist=cum if end==0 else cum[-1]-cum
    w=np.clip(1-dist/max(6.0,gap*1.5),0,1)[:,None]   # blend the snap over the first few mm
    P=P+(tgt-q)[None,:]*w
    o['P']=P;changed.add(cid)
for c in sorted(changed):
    o=D[c];r=np.asarray(o['r']);mv,mf=tube(o['P'],r,seg=12 if c.startswith(('a_','vn_')) else 10);vox,rv=voxels(o['P'],np.maximum(r,0.6))
    o['mesh']=(mv,mf);o['vox']=vox;o['radv']=rv
for x in rep:print(x)
pickle.dump(N,open('out/named_vessels.pkl','wb'));pickle.dump(S,open('out/schematic.pkl','wb'))
