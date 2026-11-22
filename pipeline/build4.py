# Population vessel atlases (Mouches & Forkert 2019): TOF average, vessel probability, vessel radius.
import numpy as np,nibabel as nib,pickle,sys
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import fast_simplification, grid
from grid import O,res
TH=float(sys.argv[1]) if len(sys.argv)>1 else 12.0   # probability threshold (%) for the 3D vessel surface
MD='../data/raw/mouches/'
grid.D=MD
# --- 1 mm TOF average for the MRA background
tof1=res('tofAverage.nii',order=1)
lo,hi=10.0,np.percentile(tof1[tof1>10],99.9)
np.save('out/mra1.npy',np.clip((tof1-lo)/(hi-lo)*255,0,255).astype(np.uint8))
# --- 0.5 mm vessel probability for the high-resolution MIP (flip x to ascending MNI)
pim=nib.load(MD+'vesselProbabilities.nii');P=pim.get_fdata(dtype=np.float32);A=pim.affine
Pf=P[::-1];x0=A[0,3]+A[0,0]*(P.shape[0]-1)   # MNI x of new index 0
nz=np.array(np.nonzero(Pf>=2));lo_i=nz.min(1);hi_i=nz.max(1)+1
crop=Pf[lo_i[0]:hi_i[0],lo_i[1]:hi_i[1],lo_i[2]:hi_i[2]]
u=np.clip(crop*2.55,0,255).astype(np.uint8)
np.save('out/mrah_u8.npy',u);np.save('out/mrah_o.npy',np.array([x0,A[1,3],A[2,3]])+lo_i*0.5)
print('mrah',u.shape,'origin',np.array([x0,A[1,3],A[2,3]])+lo_i*0.5)
# --- 3D vessel surfaces grouped as before
ter=np.load('out/ter_raw.npy')
fs=res.__globals__['nib']  # noqa
grid.D='../data/raw/dsa/human/'
fsm=res('FreeSurferSeg.nii.gz')>0
lab0=np.load('out/lab.npy')>0
brain=ndi.binary_fill_holes(ndi.binary_closing(fsm|lab0,iterations=2))
ter=np.where(brain,ter,0)
V=P>=TH
L,n=ndi.label(V);sz=np.bincount(L.ravel());keep=sz>=150;keep[0]=False;V=keep[L]
idx=np.array(np.nonzero(V)).T;mni=(np.c_[idx,np.ones(len(idx))]@A.T)[:,:3]
tfill=ter.copy();tfill[tfill>=31]=0
_,ind=ndi.distance_transform_edt(tfill==0,return_indices=True);tnear=tfill[tuple(ind)]
g=np.clip(np.round(mni-O).astype(int),0,[180,216,180]);tv=tnear[g[:,0],g[:,1],g[:,2]]
incow=(mni[:,2]<-2)&(mni[:,2]>-32)&(np.abs(mni[:,0])<26)&(mni[:,1]>-28)&(mni[:,1]<22)&~np.isin(tv,[25,26,27,28,29,30])
grp=lambda t:'ACA' if t in(1,2,3,4) else 'MCA' if 5<=t<=16 else 'PCA' if 17<=t<=24 else 'VB'
bd=ndi.binary_dilation(ndi.binary_fill_holes(ndi.binary_closing(brain,iterations=4)),iterations=4)
inb=bd[g[:,0],g[:,1],g[:,2]]
G=np.array(['SKB' if not b else ('CoW' if c else grp(t)) for c,t,b in zip(incow,tv,inb)])
side=np.where(mni[:,0]<0,'L','R')
rad1=res.__globals__  # placeholder to keep linters quiet
grid.D=MD
R1=res('vesselRadius.nii',order=0)
R1=ndi.maximum_filter(R1,3)
NM={'SKB':('Skull-base & extracranial arteries','#8c4a3c'),'CoW':('ICA & circle of Willis region','#c42a22'),'ACA':('Anterior cerebral artery branches','#f07f3c'),'MCA':('Middle cerebral artery branches','#e8463a'),'PCA':('Posterior cerebral artery branches','#d2448a'),'VB':('Vertebrobasilar & cerebellar arteries','#b04a2e')}
VS=[]
for k,(nm,col) in NM.items():
  for sd in ['L','R']:
    sel=(G==k)&(side==sd)
    if sel.sum()<60: continue
    m=np.zeros(V.shape,bool);m[tuple(idx[sel].T)]=True
    sl=ndi.find_objects(m.astype(np.uint8))[0];sl=tuple(slice(max(a.start-3,0),a.stop+3) for a in sl)
    f=ndi.gaussian_filter(m[sl].astype(np.float32),0.7)
    v,fc,_,_=marching_cubes(f,0.35)
    v=v+np.array([a.start for a in sl]);v=(np.c_[v,np.ones(len(v))]@A.T)[:,:3]
    if len(fc)>45000: v,fc=fast_simplification.simplify(v.astype(np.float32),fc.astype(np.int32),1-45000/len(fc))
    a_,b_,c_=v[fc[:,0]],v[fc[:,1]],v[fc[:,2]]
    if np.einsum('ij,ij->i',a_,np.cross(b_,c_)).sum()<0: fc=fc[:,[0,2,1]]
    q=np.unique(np.round(mni[sel]-O).astype(int),axis=0);q=q[np.all((q>=0)&(q<[181,217,181]),1)]
    flat=np.ravel_multi_index(q.T,(181,217,181)).astype(np.uint32)
    rr=R1[q[:,0],q[:,1],q[:,2]]
    VS.append(dict(id='v_'+k.lower()+'_'+sd.lower(),name=nm,group='Arteries',color=col,side=sd,n=0,vox=flat,radv=rr,mesh=(v.astype(np.float32),fc.astype(np.int32))))
    print(k,sd,int(sel.sum()),len(fc),len(flat))
old=pickle.load(open('out/vasc.pkl','rb'))
pickle.dump((VS,old[1],old[2]),open('out/vasc.pkl','wb'))
