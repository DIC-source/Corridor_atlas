import json,numpy as np,gzip,struct
from grid import *
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import fast_simplification
from tt import load_tt
OUT='out/'
# ---------- background
t1=res('human_T1w.nii.gz',order=1); t2=res('human_T2w.nii.gz',order=1)
def norm(v):
    lo,hi=np.percentile(v[v>5],[0.5,99.7]); return np.clip((v-lo)/(hi-lo)*255,0,255).astype(np.uint8)
t1=norm(t1); t2=norm(t2)
# ---------- atlases
cb=res('CerebrA.nii.gz').astype(int); at=res('ATAG_basal_ganglia.nii.gz').astype(int)
th=res('THOMAS.nii.gz').astype(int); su=res('Cerebellum-SUIT.nii.gz').astype(int)
cname={int(l.split()[0]):l.split()[1] for l in open(D+'CerebrA.txt') if l.strip()}
LOBE={'Caudal_Anterior_Cingulate':'limbic','Caudal_Middle_Frontal':'frontal','Cuneus':'occipital','Entorhinal':'limbic','Fusiform':'temporal','Inferior_Parietal':'parietal','Inferior_temporal':'temporal','Isthmus_Cingulate':'limbic','Lateral_Occipital':'occipital','Lateral_Orbitofrontal':'frontal','Lingual':'occipital','Medial_Orbitofrontal':'frontal','Middle_Temporal':'temporal','Parahippocampal':'limbic','Paracentral':'frontal','Pars_Opercularis':'frontal','Pars_Orbitalis':'frontal','Pars_Triangularis':'frontal','Pericalcarine':'occipital','Postcentral':'parietal','Posterior_Cingulate':'limbic','Precentral':'frontal','Precuneus':'parietal','Rostral_Anterior_Cingulate':'limbic','Rostral_Middle_Frontal':'frontal','Superior_Frontal':'frontal','Superior_Parietal':'parietal','Superior_Temporal':'temporal','Supramarginal':'parietal','Transverse_Temporal':'temporal','Insula':'insula'}
NICE={'Caudal_Anterior_Cingulate':'Caudal anterior cingulate','Caudal_Middle_Frontal':'Caudal middle frontal gyrus','Cuneus':'Cuneus','Entorhinal':'Entorhinal cortex','Fusiform':'Fusiform gyrus','Inferior_Parietal':'Inferior parietal lobule (angular)','Inferior_temporal':'Inferior temporal gyrus','Isthmus_Cingulate':'Isthmus of cingulate','Lateral_Occipital':'Lateral occipital cortex','Lateral_Orbitofrontal':'Lateral orbitofrontal cortex','Lingual':'Lingual gyrus','Medial_Orbitofrontal':'Medial orbitofrontal cortex','Middle_Temporal':'Middle temporal gyrus','Parahippocampal':'Parahippocampal gyrus','Paracentral':'Paracentral lobule (leg M1/S1)','Pars_Opercularis':'Pars opercularis (Broca, BA44)','Pars_Orbitalis':'Pars orbitalis','Pars_Triangularis':'Pars triangularis (Broca, BA45)','Pericalcarine':'Pericalcarine (V1, BA17)','Postcentral':'Postcentral gyrus (S1)','Posterior_Cingulate':'Posterior cingulate','Precentral':'Precentral gyrus (M1)','Precuneus':'Precuneus','Rostral_Anterior_Cingulate':'Rostral anterior cingulate','Rostral_Middle_Frontal':'Rostral middle frontal gyrus','Superior_Frontal':'Superior frontal gyrus (incl. SMA)','Superior_Parietal':'Superior parietal lobule','Superior_Temporal':'Superior temporal gyrus (post. = Wernicke)','Supramarginal':'Supramarginal gyrus','Transverse_Temporal':'Heschl gyrus (A1)','Insula':'Insula'}
ELOQ={'Precentral','Postcentral','Paracentral','Pars_Opercularis','Pars_Triangularis','Superior_Temporal','Supramarginal','Pericalcarine','Transverse_Temporal'}
LCOL={'frontal':'#e9b8a3','parietal':'#efd9a0','temporal':'#bcd7a8','occipital':'#a9c7de','limbic':'#cdb8dc','insula':'#e8c27f'}
ECOL={'Precentral':'#d9665b','Postcentral':'#5f8fcf','Paracentral':'#c97a8a','Pars_Opercularis':'#e0913a','Pars_Triangularis':'#ecab55','Superior_Temporal':'#6fae5f','Supramarginal':'#d2b84a','Pericalcarine':'#3f78b5','Transverse_Temporal':'#3e9a7c'}
S=[]  # structures
lab=np.zeros(cb.shape,np.uint8)
def add(mask,name,group,color,side='',note=''):
    i=len(S)+1; S.append(dict(id=i,name=name,group=group,color=color,side=side,note=note)); lab[mask]=i; return i
for side,pre in [('L','Left_'),('R','Right_')]:
    for k,v in cname.items():
        n=v.replace('Left_','').replace('Right_','')
        if v.startswith(pre) and n in LOBE:
            add(cb==k,NICE[n],'Cortex · '+LOBE[n],ECOL.get(n,LCOL[LOBE[n]]),side,'eloquent' if n in ELOQ else '')
inv={v:k for k,v in cname.items()}
g=lambda n:(cb==inv['Left_'+n])|(cb==inv['Right_'+n])
sideL=X<0
# ventricles (merged midline)
add(g('Lateral_Ventricle')|g('Inferior_Lateral_Ventricle'),'Lateral ventricle','Ventricles','#7fb6d9')
add(g('Third_Ventricle'),'Third ventricle','Ventricles','#6aa5cf')
add(g('Fourth_Ventricle'),'Fourth ventricle','Ventricles','#5a97c4')
# brainstem split (approximate axial planes in MNI)
bs=g('Brainstem')
add(bs&(Z>=-18),'Midbrain','Brainstem','#c9a27e',note='approx. boundary z ≥ −18')
add(bs&(Z<-18)&(Z>=-44),'Pons','Brainstem','#bb9070',note='approx. boundary −44 ≤ z < −18')
add(bs&(Z<-44),'Medulla','Brainstem','#ad8066',note='approx. boundary z < −44')
add(g('Optic_Chiasm'),'Optic chiasm','Diencephalon','#f0d35a')
for side,pre in [('L','Left_'),('R','Right_')]:
    c=lambda n:cb==inv[pre+n]
    add(c('Ventral_Diencephalon'),'Ventral diencephalon (hypothalamus/subthalamus)','Diencephalon','#d8a7b5',side)
    add(c('Basal_Forebrain'),'Basal forebrain','Basal ganglia','#c7b49a',side)
    add(c('Caudate'),'Caudate nucleus','Basal ganglia','#7c9fc9',side)
    add(c('Putamen'),'Putamen','Basal ganglia','#d89a9a',side)
    add(c('Accumbens_Area'),'Nucleus accumbens','Basal ganglia','#a58fc2',side)
    pal=c('Pallidum'); o=1 if side=='L' else 2
    add(pal,'Globus pallidus externus (GPe)','Basal ganglia','#9fbf8a',side)
    add(pal&(at==10+o-0) if False else pal&(at==(11 if side=='L' else 12)),'Globus pallidus internus (GPi)','Basal ganglia','#6f9e5a',side)
    add(c('Hippocampus'),'Hippocampus','Limbic','#e3c06c',side)
    add(c('Amygdala'),'Amygdala','Limbic','#d78f62',side)
    add(c('Thalamus'),'Thalamus (other nuclei)','Thalamus','#b69bc8',side)
    TH=[('AV',1,'Anterior nucleus (AV)','#e7b4d2'),('VA',3,'Ventral anterior (VA)','#c79ad7'),('VLa',5,'Ventral lateral anterior (VLa)','#a77bd0'),('VLp',7,'Ventral lateral posterior (VLp / Vim)','#8a5fc4'),('VPL',9,'Ventral posterolateral (VPL)','#6f7fd0'),('Pul',11,'Pulvinar','#9cb0dd'),('LGN',13,'Lateral geniculate (LGN)','#4f86c6'),('MGN',15,'Medial geniculate (MGN)','#45a39a'),('CM',17,'Centromedian (CM)','#d07fae'),('MD',19,'Mediodorsal (MD)','#c4a3e6')]
    for _,k,n,col in TH:
        add(th==k+(0 if side=='L' else 1),n,'Thalamus',col,side)
    add(at==(1 if side=='L' else 2),'Red nucleus','Brainstem nuclei','#c0392b',side)
    add(at==(3 if side=='L' else 4),'Substantia nigra','Brainstem nuclei','#4a3b36',side)
    cx=-9.8 if side=='L' else 10.2
    add(((X-cx)/3.2)**2+((Y+12.5)/5.5)**2+((Z+6)/2.6)**2<=1,'Subthalamic nucleus (STN)','Brainstem nuclei','#e07b39',side,'approximate ellipsoid from ATAG centroid')
# cerebellum
cbl=g('Cerebellum_Gray_Matter')|g('Cerebellum_White_Matter')|g('Vermal_lobules_I-V')|g('Vermal_lobules_VI-VII')|g('Vermal_lobules_VIII-X')
add(cbl,'Cerebellar white matter / dentate region','Cerebellum','#d9c7a8')
grp={'Anterior lobe (I–V)':[1,3],'Posterior lobe (VI–VIII)':[5,8,11,14,17,20],'Tonsil / IX':[23],'Flocculus (X)':[26]}
for side,dd in [('L',0),('R',1)]:
    for n,ks in grp.items():
        m=np.isin(su,[k+dd*(1 if k in(1,3) else 2) for k in ks])
        add(m&cbl,'Cerebellum · '+n,'Cerebellum',{'Anterior lobe (I–V)':'#e3b38a','Posterior lobe (VI–VIII)':'#d9a27a','Tonsil / IX':'#c98a6a','Flocculus (X)':'#b9785c'}[n],side)
add(np.isin(su,[6,9,12,15,18,21,24,27])&cbl,'Vermis','Cerebellum','#c4865e')
print('structures',len(S))
cnt=np.bincount(lab.ravel(),minlength=len(S)+1)
for s in S: s['vox']=int(cnt[s['id']])
print([ (s['name'],s['side'],s['vox']) for s in S if s['vox']<60])
np.save(OUT+'lab.npy',lab); np.save(OUT+'t1.npy',t1); np.save(OUT+'t2.npy',t2)
json.dump(S,open(OUT+'S.json','w'))
