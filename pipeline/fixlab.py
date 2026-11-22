# Anatomical corrections around the diencephalon-midbrain junction
import numpy as np,json
from scipy import ndimage as ndi
from tt import load_tt
O=np.array([-90,-126,-72]);lab=np.load('out/lab_v1.npy');S=json.load(open('out/S_v1.json'))
X,Y,Z=np.meshgrid(*[O[i]+np.arange(lab.shape[i]) for i in range(3)],indexing='ij')
byname=lambda n,sd:[s for s in S if s['name']==n and s['side']==sd][0]
def add(name,group,color,side,note=''):
    i=len(S)+1;S.append(dict(id=i,name=name,group=group,color=color,side=side,note=note));return i
# 1) mesencephalon side of the plane posterior commissure (0,-26,0) -> posterior mammillary body (0,-10,-14)
meso=((Y+26)*(-14)-(Z-0)*16)>0
tracks,cl,names=load_tt()
cst=np.zeros(lab.shape,bool)
for i in np.nonzero(np.isin(cl,[62,63,64,65,66,67,68,69,70,71]))[0]:   # CST, corticobulbar, corticopontine
    v=np.round(tracks[i]-O).astype(int);ok=np.all((v>=0)&(v<lab.shape),1);v=v[ok];cst[v[:,0],v[:,1],v[:,2]]=True
cst=ndi.binary_dilation(cst,iterations=1)
mid=byname('Midbrain','')['id']
for sd in ['L','R']:
    vdc=byname('Ventral diencephalon (hypothalamus/subthalamus)',sd)
    m=(lab==vdc['id'])&meso
    crus=m&cst
    cid=add('Cerebral peduncle (crus cerebri)','Brainstem','#d4b48c',sd,'CST, corticobulbar and corticopontine fibres')
    lab[crus]=cid
    teg=m&~crus&(np.abs(X)<=15)               # medial part = midbrain tegmentum
    lab[teg]=mid
    lat=m&~crus&~teg                           # lateral leftovers: optic tract / geniculate region at the junction
    if lat.any():
        lid=add('Diencephalon–midbrain junction (lateral)','Diencephalon','#cfa0a8',sd,'optic tract / geniculate region; boundary approximate');lab[lat]=lid
    vdc['name']='Hypothalamus / subthalamic region'
    print(sd,'ventral DC -> crus',int(crus.sum()),'-> midbrain',int((m&~crus).sum()))
# 2) STN centred on the DISTAL-atlas centroid (about ±12, -13, -5.5), oriented ellipsoid
for sd,m_ in [('L',-1),('R',1)]:
    s=byname('Subthalamic nucleus (STN)',sd);lab[lab==s['id']]=0
    # restore what was underneath (ventral DC / hypothalamus region)
    old=np.load('out/lab_v1.npy');lab[(old==s['id'])&(lab==0)]=byname('Hypothalamus / subthalamic region',sd)['id']
    e=((X-12*m_)/3.0)**2+((Y+13)/5.0)**2+((Z+5.5)/2.5)**2<=1
    lab[e]=s['id'];s['note']='approximate ellipsoid at the DISTAL STN centroid (±12, −13, −5.5)'
# 3) substantia nigra: the ATAG label is thin at 1 mm; grow it 1 voxel inside the ventral midbrain
for sd in ['L','R']:
    s=byname('Substantia nigra',sd);m=lab==s['id']
    grow=ndi.binary_dilation(m,iterations=1)&np.isin(lab,[mid]+[q['id'] for q in S if q['name'].startswith('Cerebral peduncle') and q['side']==sd]+[byname('Hypothalamus / subthalamic region',sd)['id']])
    lab[grow]=s['id'];print(sd,'SN',int(m.sum()),'->',int((lab==s['id']).sum()))
cnt=np.bincount(lab.ravel(),minlength=len(S)+1)
for s in S: s['vox']=int(cnt[s['id']])
np.save('out/lab.npy',lab);json.dump(S,open('out/S.json','w'))
