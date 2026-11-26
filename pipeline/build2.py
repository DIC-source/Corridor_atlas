import json,numpy as np
from grid import O
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import fast_simplification
from tt import load_tt
OUT='out/'
lab=np.load(OUT+'lab.npy'); S=json.load(open(OUT+'S.json'))
# ---------- meshes
objs=ndi.find_objects(lab)
MESH=[]
for s in S:
    sl=objs[s['id']-1]
    if sl is None: MESH.append(None); continue
    sl=tuple(slice(max(a.start-3,0),a.stop+3) for a in sl)
    m=(lab[sl]==s['id']).astype(np.float32)
    small=s['vox']<400
    m=ndi.gaussian_filter(m,0.6 if small else 0.9)
    try: v,f,_,_=marching_cubes(m,0.35 if small else 0.45)
    except Exception as e: print('fail',s['name']); MESH.append(None); continue
    v=v+np.array([a.start for a in sl])+O
    tgt=1500 if small else (3500 if 'Cortex' in s['group'] else 2500)
    if len(f)>tgt:
        v,f=fast_simplification.simplify(v.astype(np.float32),f.astype(np.int32),1-tgt/len(f))
    MESH.append((v.astype(np.float32),f.astype(np.int32)))
print('mesh tris',sum(len(m[1]) for m in MESH if m))
# ---------- tracts
tracks,cl,names=load_tt()
T=[ # id,name,group,color,cluster ids
('cst_l','Corticospinal tract','Motor','#2f62d0',[62],'L'),('cst_r','Corticospinal tract','Motor','#2f62d0',[63],'R'),
('cbt_l','Corticobulbar tract','Motor','#5d8be8',[64],'L'),('cbt_r','Corticobulbar tract','Motor','#5d8be8',[65],'R'),
('fat_l','Frontal aslant tract','Motor','#7b5cd6',[16],'L'),('fat_r','Frontal aslant tract','Motor','#7b5cd6',[17],'R'),
('ml_l','Medial lemniscus','Sensory','#1f9e8f',[72],'L'),('ml_r','Medial lemniscus','Sensory','#1f9e8f',[73],'R'),
('str_l','Superior thalamic radiation','Sensory','#38b6a6',[46],'L'),('str_r','Superior thalamic radiation','Sensory','#38b6a6',[49],'R'),
('af_l','Arcuate fasciculus','Language','#d9442e',[0],'L'),('af_r','Arcuate fasciculus','Language','#d9442e',[1],'R'),
('slf_l','SLF II–III','Language','#ef8a2f',[28,29],'L'),('slf_r','SLF II–III','Language','#ef8a2f',[30,31],'R'),
('ifof_l','Inferior fronto-occipital fasciculus','Language','#c2389b',[20],'L'),('ifof_r','Inferior fronto-occipital fasciculus','Language','#c2389b',[21],'R'),
('unc_l','Uncinate fasciculus','Language','#e6b422',[32],'L'),('unc_r','Uncinate fasciculus','Language','#e6b422',[33],'R'),
('ilf_l','Inferior longitudinal fasciculus','Language','#8fae2c',[22],'L'),('ilf_r','Inferior longitudinal fasciculus','Language','#8fae2c',[23],'R'),
('or_l','Optic radiation (geniculocalcarine)','Visual','#1d7fd1',[50],'L'),('or_r','Optic radiation (geniculocalcarine)','Visual','#1d7fd1',[51],'R'),
('cn2_l','Optic nerve / tract (CN II)','Visual','#f2c230',[96],'L'),('cn2_r','Optic nerve / tract (CN II)','Visual','#f2c230',[97],'R'),
('ar_l','Acoustic radiation','Auditory','#22a37a',[36],'L'),('ar_r','Acoustic radiation','Auditory','#22a37a',[37],'R'),
('cing_l','Cingulum','Limbic','#a855c7',[2,3,4,5,6],'L'),('cing_r','Cingulum','Limbic','#a855c7',[8,9,10,11,12],'R'),
('fx_l','Fornix','Limbic','#e070a8',[52],'L'),('fx_r','Fornix','Limbic','#e070a8',[53],'R'),
('atr_l','Anterior thalamic radiation','Limbic','#b98ad8',[44],'L'),('atr_r','Anterior thalamic radiation','Limbic','#b98ad8',[47],'R'),
('cc_minor','Corpus callosum · forceps minor','Commissural','#d6453b',[85],''),('cc_body','Corpus callosum · body','Commissural','#e0604f',[86],''),
('cc_tap','Corpus callosum · tapetum','Commissural','#c94e6e',[87],''),('cc_major','Corpus callosum · forceps major','Commissural','#b8404d',[88],''),
('ac','Anterior commissure','Commissural','#f08a5d',[82,83,84],''),
('cpt_l','Corticopontine tracts','Brainstem & cerebellar','#7a8fa8',[66,67,68],'L'),('cpt_r','Corticopontine tracts','Brainstem & cerebellar','#7a8fa8',[69,70,71],'R'),
('drtt','Dentato-rubro-thalamic tract','Brainstem & cerebellar','#d6702a',[74,75,76,77],''),
('scp','Superior cerebellar peduncle','Brainstem & cerebellar','#c4572b',[94],''),('mcp','Middle cerebellar peduncle','Brainstem & cerebellar','#a1887f',[93],''),
('icp_l','Inferior cerebellar peduncle','Brainstem & cerebellar','#8d6e63',[91],'L'),('icp_r','Inferior cerebellar peduncle','Brainstem & cerebellar','#8d6e63',[92],'R'),
('cn3_l','Oculomotor nerve (CN III)','Cranial nerves','#f4a340',[98],'L'),('cn3_r','Oculomotor nerve (CN III)','Cranial nerves','#f4a340',[99],'R'),
('cn5_l','Trigeminal nerve (CN V)','Cranial nerves','#e8c547',[100],'L'),('cn5_r','Trigeminal nerve (CN V)','Cranial nerves','#e8c547',[101],'R'),
('cn7_l','Facial nerve (CN VII)','Cranial nerves','#d98c3a',[102],'L'),('cn7_r','Facial nerve (CN VII)','Cranial nerves','#d98c3a',[103],'R'),
('cn8_l','Vestibulocochlear nerve (CN VIII)','Cranial nerves','#c9a43a',[104],'L'),('cn8_r','Vestibulocochlear nerve (CN VIII)','Cranial nerves','#c9a43a',[105],'R'),
]
rng=np.random.default_rng(1)
TR=[]
shape=lab.shape
for tid,name,grp,col,ids,side in T:
    idx=np.nonzero(np.isin(cl,ids))[0]
    if len(idx)==0: print('empty',tid); continue
    # voxel mask from all streamlines
    msk=np.zeros(shape,bool)
    for i in idx:
        p=tracks[i]
        # densify
        seg=np.linspace(0,1,3)[:-1]
        q=np.vstack([p[:-1]+(p[1:]-p[:-1])*t for t in seg]+[p[-1:]])
        v=np.round(q-O).astype(int); ok=np.all((v>=0)&(v<np.array(shape)),1); v=v[ok]
        msk[v[:,0],v[:,1],v[:,2]]=True
    msk=ndi.binary_closing(msk,iterations=1)|msk
    vox=np.nonzero(msk.ravel())[0].astype(np.uint32)
    sel=idx if len(idx)<=70 else rng.choice(idx,70,replace=False)
    lines=[]
    for i in sel:
        p=tracks[i]; p=p[::max(1,len(p)//60)] if len(p)>60 else p
        lines.append(np.round(p*4).astype(np.int16))
    TR.append(dict(id=tid,name=name,group=grp,color=col,side=side,n=int(len(idx)),vox=vox,lines=lines))
    print(tid,len(idx),len(vox))
import pickle; pickle.dump((MESH,TR,T),open(OUT+'geo.pkl','wb'))
