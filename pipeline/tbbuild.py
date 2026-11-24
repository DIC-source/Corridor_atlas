# Reference subject (TopBrain 008, MRA arteries + CTA deep veins) warped to MNI -> meshes, voxels, point clouds
import numpy as np,nibabel as nib,json,pickle
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import fast_simplification
SID='008';O=np.array([-90,-126,-72]);SH=(181,217,181)
im=nib.load(f'out/tb/{SID}_mr_lab.nii.gz');A=im.affine;M=np.asanyarray(im.dataobj)
C=np.asanyarray(nib.load(f'out/tb/{SID}_ct_lab.nii.gz').dataobj)
lab2={v:k for k,v in json.load(open('../data/raw/topbrain/labels_topbrain_v2_topaneu36class.json'))['labels'].items()}
NAME={'BA':'Basilar artery','P1P2':'PCA P1–P2','ICA':'Internal carotid artery','M1':'MCA M1 (sphenoidal)','Pcom':'Posterior communicating artery','Acom':'Anterior communicating artery',
 'A1A2':'ACA A1–A2','A3':'ACA A3 (pericallosal)','3rd-A2':'Median (third) A2','3rd-A3':'Median (third) A3','M2':'MCA M2 (insular)','M3':'MCA M3 (opercular)','P3P4':'PCA P3–P4',
 'VA':'Vertebral artery (V4)','SCA':'Superior cerebellar artery','AICA':'Anterior inferior cerebellar artery (AICA)','PICA':'Posterior inferior cerebellar artery (PICA)',
 'AChA':'Anterior choroidal artery','OA':'Ophthalmic artery'}
COL={'BA':'#9b2d20','P1P2':'#c43c7a','ICA':'#c0392b','M1':'#e2463a','Pcom':'#d94b6e','Acom':'#e8742f','A1A2':'#e8742f','A3':'#f08b3c','3rd-A2':'#f08b3c','3rd-A3':'#f08b3c','M2':'#ee6a50','M3':'#f28a73',
 'P3P4':'#b5487f','VA':'#8a3324','SCA':'#a8493a','AICA':'#b65a45','PICA':'#a0503a','AChA':'#d0604e','OA':'#d46a5a'}
groups={}
for v,k in lab2.items():
    if v==0:continue
    side='R' if k.startswith('R-') else 'L' if k.startswith('L-') else ''
    base=k[2:] if side else k;base='ICA' if base.startswith('ICA') else base
    groups.setdefault((base,side),[]).append(v)
def mesh_of(m):
    sl=ndi.find_objects(m.astype(np.uint8))[0];sl=tuple(slice(max(a.start-3,0),a.stop+3) for a in sl)
    f=ndi.gaussian_filter(m[sl].astype(np.float32),0.8);v,fc,_,_=marching_cubes(f,0.4)
    v=v+np.array([a.start for a in sl]);v=(np.c_[v,np.ones(len(v))]@A.T)[:,:3]
    if len(fc)>18000:v,fc=fast_simplification.simplify(v.astype(np.float32),fc.astype(np.int32),1-18000/len(fc))
    a_,b_,c_=v[fc[:,0]],v[fc[:,1]],v[fc[:,2]]
    if np.einsum('ij,ij->i',a_,np.cross(b_,c_)).sum()<0:fc=fc[:,[0,2,1]]
    return v.astype(np.float32),fc.astype(np.int32)
def entry(id_,name,group,color,side,m,src):
    idx=np.array(np.nonzero(m)).T;mni=(np.c_[idx,np.ones(len(idx))]@A.T)[:,:3]
    dt=ndi.distance_transform_edt(m)*0.5;rad=dt[tuple(idx.T)]
    q=np.round(mni-O).astype(int);ok=np.all((q>=0)&(q<SH),1);flat=np.ravel_multi_index(q[ok].T,SH)
    u,inv=np.unique(flat,return_inverse=True);rv=np.zeros(len(u));np.maximum.at(rv,inv,rad[ok])
    P=mni[::max(1,len(mni)//1500)]
    return dict(id=id_,name=name,group=group,color=color,side=side,vox=u.astype(np.uint32),radv=np.clip(rv,.5,4),mesh=mesh_of(m),P=P,r=None,tb=True,src=src)
OUT=[]
for (base,side),vals in sorted(groups.items()):
    m=np.isin(M,vals)
    src='TopBrain case '+SID+' MRA annotation'
    if m.sum()<20:
        # missing on this side in the reference case: mirror the other side
        other=groups.get((base,'R' if side=='L' else 'L'))
        if not other:continue
        mm=np.isin(M,other)
        if mm.sum()<20:continue
        # mirror across MNI x=0: index i -> mirrored index
        x0=A[0,3];dx=A[0,0];idx=np.arange(M.shape[0]);xm=-(x0+dx*idx);ii=np.round((xm-x0)/dx).astype(int)
        ok=(ii>=0)&(ii<M.shape[0]);m=np.zeros_like(mm);m[idx[ok]]=mm[ii[ok]];src='mirrored from the other side (missing in case '+SID+')'
    nm=NAME.get(base,base);k=('tb_'+base.lower().replace('-','')+('_'+side.lower() if side else ''))
    OUT.append(entry(k,nm,'Arteries',COL.get(base,'#c0392b'),side,m,src));print(k,int(m.sum()),src)
VN={35:('vog','Great cerebral vein (Galen)',''),36:('sts','Straight sinus',''),37:('icv','Internal cerebral veins','')}
for v,(k,nm,sd) in VN.items():
    m=C==v
    if m.sum()<20:continue
    OUT.append(entry('tbv_'+k,nm,'Veins & sinuses','#3566b8' if 'sinus' in nm else '#4b86d4',sd,m,'TopBrain case '+SID+' CTA annotation'));print(k,int(m.sum()))
pickle.dump(OUT,open('out/topbrain.pkl','wb'))
