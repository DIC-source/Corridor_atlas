import numpy as np,nibabel as nib,pickle,json
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import fast_simplification
from grid import O
R='../data/raw/'
ter=np.load('out/ter_raw.npy')  # 1mm grid
import grid
fs=grid.res('FreeSurferSeg.nii.gz')>0
lab0=np.load('out/lab.npy')>0
brain=ndi.binary_fill_holes(ndi.binary_closing(fs|lab0,iterations=2))
ter=np.where(brain,ter,0)
vh=nib.load('out/vessel_mni.nii.gz'); V=vh.get_fdata()>0.5; Ah=vh.affine  # 0.5mm grid, origin -98,-134,-72
cow=np.asanyarray(nib.load(R+'HQA_Cerebrovasculature/3_feature_extraction/atlas/CoW_atlas.nii.gz').dataobj)>0  # 0.5mm origin -90,-126,-72
# remove tiny islands
lab,n=ndi.label(V); sz=np.bincount(lab.ravel()); keep=sz>=40; keep[0]=False; V=keep[lab]
idx=np.array(np.nonzero(V)).T; mni=idx*0.5+np.array([-98,-134,-72])
# territory per vessel voxel (nearest territory)
tfill=ter.copy(); tfill[tfill>=31]=0
_,ind=ndi.distance_transform_edt(tfill==0,return_indices=True); tnear=tfill[tuple(ind)]
g=np.clip(np.round(mni-O).astype(int),0,[180,216,180]); tv=tnear[g[:,0],g[:,1],g[:,2]]
ci=np.round((mni-np.array([-90,-126,-72]))/0.5).astype(int); okc=np.all((ci>=0)&(ci<np.array(cow.shape)),1)
incow=(mni[:,2]<-2)&(np.abs(mni[:,0])<26)&(mni[:,1]>-28)&(mni[:,1]<22)&~np.isin(tv,[25,26,27,28,29,30])
def grp(t):
    if t in(1,2,3,4):return 'ACA'
    if 5<=t<=16:return 'MCA'
    if 17<=t<=24:return 'PCA'
    return 'VB'
bd=ndi.binary_dilation(ndi.binary_fill_holes(ndi.binary_closing(brain,iterations=4)),iterations=4)
inb=bd[g[:,0],g[:,1],g[:,2]]
incow=incow&(mni[:,2]>-32)
G=np.array(['SKB' if not b else ('CoW' if c else grp(t)) for c,t,b in zip(incow,tv,inb)])
side=np.where(mni[:,0]<0,'L','R')
NM={'SKB':('Skull-base & extracranial arteries','#8c4a3c'),'CoW':('ICA & circle of Willis region','#c42a22'),'ACA':('Anterior cerebral artery branches','#f07f3c'),'MCA':('Middle cerebral artery branches','#e8463a'),'PCA':('Posterior cerebral artery branches','#d2448a'),'VB':('Vertebrobasilar & cerebellar arteries','#b04a2e')}
VS=[]
for k,(nm,col) in NM.items():
    for sd in (['L','R']):
        sel=(G==k)&(side==sd)
        if sel.sum()<30: continue
        m=np.zeros(V.shape,bool); m[tuple(idx[sel].T)]=True
        sl=ndi.find_objects(m.astype(np.uint8))[0]; sl=tuple(slice(max(a.start-3,0),a.stop+3) for a in sl)
        f=ndi.gaussian_filter(m[sl].astype(np.float32),0.6)
        v,fc,_,_=marching_cubes(f,0.3)
        v=(v+np.array([a.start for a in sl]))*0.5+np.array([-98,-134,-72])
        if len(fc)>40000: v,fc=fast_simplification.simplify(v.astype(np.float32),fc.astype(np.int32),1-40000/len(fc))
        # 1mm voxels for MRI/proximity
        q=np.unique(np.round(mni[sel]-O).astype(int),axis=0)
        flat=np.ravel_multi_index(q.T,(181,217,181)).astype(np.uint32)
        VS.append(dict(id='v_'+k.lower()+'_'+sd.lower(),name=nm,group='Arteries',color=col,side=sd,n=0,vox=flat,mesh=(v.astype(np.float32),fc.astype(np.int32))))
        print(k,sd,sel.sum(),len(fc),len(flat))
# territories
TN=['ACA','ACA','Medial lenticulostriate (ACA)','Medial lenticulostriate (ACA)','Lateral lenticulostriate (MCA)','Lateral lenticulostriate (MCA)','MCA · frontal','MCA · frontal','MCA · parietal','MCA · parietal','MCA · temporal','MCA · temporal','MCA · occipital','MCA · occipital','MCA · insular','MCA · insular','PCA · temporal','PCA · temporal','PCA · occipital','PCA · occipital','Posterior choroidal & thalamoperforators','Posterior choroidal & thalamoperforators','Anterior choroidal & thalamoperforators','Anterior choroidal & thalamoperforators','Basilar (brainstem perforators)','Basilar (brainstem perforators)','Superior cerebellar artery','Superior cerebellar artery','Inferior cerebellar arteries (AICA/PICA)','Inferior cerebellar arteries (AICA/PICA)']
TC=['#7fbf6a','#7fbf6a','#c8d94a','#c8d94a','#f2a541','#f2a541','#5b8fd6','#5b8fd6','#4fb6c9','#4fb6c9','#7a6fd1','#7a6fd1','#3e6fb0','#3e6fb0','#58a9e8','#58a9e8','#c76fb4','#c76fb4','#a65a9e','#a65a9e','#e07a8f','#e07a8f','#f0b37e','#f0b37e','#b98a5e','#b98a5e','#d1a073','#d1a073','#a7754a','#a7754a']
TG=lambda k:'Territory · ACA' if k<=4 else 'Territory · MCA' if k<=16 else 'Territory · PCA' if k<=24 else 'Territory · vertebrobasilar'
TE=[];terv=np.zeros(ter.shape,np.uint8)
objs=ndi.find_objects(np.where(ter<=30,ter,0))
for k in range(1,31):
    m=ter==k; s='L' if k%2 else 'R'
    # verify side from centroid
    cx=np.nonzero(m)[0].mean()+O[0]; s='L' if cx<0 else 'R'
    terv[m]=k
    sl=tuple(slice(max(a.start-2,0),a.stop+2) for a in objs[k-1])
    f=ndi.gaussian_filter(m[sl].astype(np.float32),1.0); v,fc,_,_=marching_cubes(f,0.5)
    v=v+np.array([a.start for a in sl])+O
    if len(fc)>3000: v,fc=fast_simplification.simplify(v.astype(np.float32),fc.astype(np.int32),1-3000/len(fc))
    TE.append(dict(id=k,name=TN[k-1]+' territory',group=TG(k),color=TC[k-1],side=s,mesh=(v.astype(np.float32),fc.astype(np.int32))))
pickle.dump((VS,TE,terv),open('out/vasc.pkl','wb'))
