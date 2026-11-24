# Population (generic) named vessels: Mouches population vessel surface, segmented by the TopBrain 5-case consensus of vessel names.
import numpy as np,nibabel as nib,json,pickle
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import fast_simplification
CASES=['005','006','007','008','010'];O=np.array([-90,-126,-72]);SH=(181,217,181);HO=np.array([-91.25,-126.25,-72.25])
A=np.diag([.5,.5,.5,1.]);A[:3,3]=HO
lab2={v:k for k,v in json.load(open('../data/raw/topbrain/labels_topbrain_v2_topaneu36class.json'))['labels'].items()}
def to1mm(onehot):   # 0.5 mm (ascending x) -> 1 mm MNI grid, block mean
    X,Y,Z=onehot.shape;out=np.zeros(SH,np.float32)
    a=onehot[2:2+2*SH[0]];a=a[:(a.shape[0]//2)*2,:(Y//2)*2,:(Z//2)*2].reshape(a.shape[0]//2,2,Y//2,2,Z//2,2).mean((1,3,5))
    s=[min(SH[i],a.shape[i]) for i in range(3)];out[:s[0],:s[1],:s[2]]=a[:s[0],:s[1],:s[2]];return out
MR=[np.asanyarray(nib.load(f'out/tb/{c}_mr_lab2.nii.gz').dataobj)[::-1] for c in CASES]
CT=[np.asanyarray(nib.load(f'out/tb/{c}_ct_lab.nii.gz' if False else f'out/tb/{c}_ct_lab2.nii.gz').dataobj)[::-1] for c in CASES]
groups={}
for v,k in lab2.items():
    if v==0 or k.startswith('3rd'):continue          # median A2/A3 is an anatomical variant: left out of the generic atlas
    side='R' if k.startswith('R-') else 'L' if k.startswith('L-') else ''
    base=k[2:] if side else k;base='ICA' if base.startswith('ICA') else base
    groups.setdefault((base,side),[]).append(v)
keys=sorted(groups);best=np.zeros(SH,np.float32);arg=np.full(SH,-1,np.int16);Pc={}
for n,(base,side) in enumerate(keys):
    p=np.mean([to1mm(np.isin(L,groups[(base,side)]).astype(np.float32)) for L in MR],0)
    Pc[(base,side)]=p;ps=ndi.gaussian_filter(p,3.0);m=ps>best;best[m]=ps[m];arg[m]=n
print('class maps done')
P=np.load('out/Ph.npy');Vm=P>=25
lab_,nc=ndi.label(Vm);sz=np.bincount(lab_.ravel());keep=sz>=150;keep[0]=False;Vm=keep[lab_]
idx=np.array(np.nonzero(Vm)).T;i1=np.clip((idx-[2,0,0])//2,0,np.array(SH)-1)
cls=np.where(best[tuple(i1.T)]>=0.03,arg[tuple(i1.T)],-1)
NAME={'BA':'Basilar artery','P1P2':'PCA P1–P2','ICA':'Internal carotid artery','M1':'MCA M1 (sphenoidal)','Pcom':'Posterior communicating artery','Acom':'Anterior communicating artery',
 'A1A2':'ACA A1–A2','A3':'ACA A3 (pericallosal)','M2':'MCA M2 (insular)','M3':'MCA M3 (opercular)','P3P4':'PCA P3–P4','VA':'Vertebral artery (V4)','SCA':'Superior cerebellar artery',
 'AICA':'Anterior inferior cerebellar artery (AICA)','PICA':'Posterior inferior cerebellar artery (PICA)','AChA':'Anterior choroidal artery','OA':'Ophthalmic artery'}
COL={'BA':'#9b2d20','P1P2':'#c43c7a','ICA':'#c0392b','M1':'#e2463a','Pcom':'#d94b6e','Acom':'#e8742f','A1A2':'#e8742f','A3':'#f08b3c','M2':'#ee6a50','M3':'#f28a73',
 'P3P4':'#b5487f','VA':'#8a3324','SCA':'#a8493a','AICA':'#b65a45','PICA':'#a0503a','AChA':'#d0604e','OA':'#d46a5a'}
def mesh_of(m,Aff,sig=0.8,lvl=0.4,cap=7000):
    sl=ndi.find_objects(m.astype(np.uint8))[0];sl=tuple(slice(max(a.start-3,0),a.stop+3) for a in sl)
    f=ndi.gaussian_filter(m[sl].astype(np.float32),sig);v,fc,_,_=marching_cubes(f,lvl)
    v=v+np.array([a.start for a in sl]);v=(np.c_[v,np.ones(len(v))]@Aff.T)[:,:3]
    if len(fc)>cap:v,fc=fast_simplification.simplify(v.astype(np.float32),fc.astype(np.int32),1-cap/len(fc))
    a_,b_,c_=v[fc[:,0]],v[fc[:,1]],v[fc[:,2]]
    if np.einsum('ij,ij->i',a_,np.cross(b_,c_)).sum()<0:fc=fc[:,[0,2,1]]
    return v.astype(np.float32),fc.astype(np.int32)
Rad=nib.load('../data/raw/mouches/vesselRadius.nii').get_fdata(dtype=np.float32)[::-1]
def entry(id_,name,group,color,side,mni,rad,mesh,src):
    q=np.round(mni-O).astype(int);ok=np.all((q>=0)&(q<SH),1);flat=np.ravel_multi_index(q[ok].T,SH)
    u,inv=np.unique(flat,return_inverse=True);rv=np.zeros(len(u));np.maximum.at(rv,inv,rad[ok])
    return dict(id=id_,name=name,group=group,color=color,side=side,vox=u.astype(np.uint32),radv=np.clip(rv,.5,4),mesh=mesh,P=mni[::max(1,len(mni)//1500)],r=None,tb=True,src=src)
OUT=[];A1=np.diag([1.,1,1,1]);A1[:3,3]=O
for n,(base,side) in enumerate(keys):
    sel=cls==n;src='population vessel surface (Mouches & Forkert) named by the TopBrain 5-case consensus'
    nm=NAME.get(base,base);k='tb_'+base.lower()+('_'+side.lower() if side else '')
    if sel.sum()>=150:
        m=np.zeros(P.shape,bool);m[tuple(idx[sel].T)]=True
        mni=idx[sel]*0.5+HO;rad=Rad[tuple(idx[sel].T)];OUT.append(entry(k,nm,'Arteries',COL.get(base,'#c0392b'),side,mni,rad,mesh_of(m,A),src))
    else:
        print('left to the schematic layer',k,int(sel.sum()));continue
    print(k,int(sel.sum()),OUT[-1]['src'][:22])
VN={35:('vog','Great cerebral vein (Galen)'),36:('sts','Straight sinus'),37:('icv','Internal cerebral veins')}
for v,(k,nm) in VN.items():
    p=ndi.gaussian_filter(np.mean([to1mm((L==v).astype(np.float32)) for L in CT],0),1.0);m=p>=0.25
    lb,nn=ndi.label(m);m=lb==(np.argmax(np.bincount(lb.ravel())[1:])+1)
    mi=np.array(np.nonzero(m)).T;mni=(mi+O).astype(float)
    OUT.append(entry('tbv_'+k,nm,'Veins & sinuses','#3566b8' if 'sinus' in nm else '#4b86d4','',mni,np.full(len(mni),2.0),mesh_of(m,A1,sig=0.8,lvl=0.4),'TopBrain 5-case CTA consensus'));print(k,int(m.sum()),mni.mean(0).round(0))
pickle.dump(OUT,open('out/topbrain.pkl','wb'))
