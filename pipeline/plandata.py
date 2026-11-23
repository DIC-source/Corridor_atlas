# Pre-op planning layers: distance-to-vessel, distance-to-sulcus, critical-tract direction field, entry candidates
import numpy as np,json,pickle,gzip
from scipy import ndimage as ndi
from tt import load_tt
O=np.array([-90,-126,-72]);SH=(181,217,181)
t1=np.load('out/t1.npy');lab=np.load('out/lab.npy');S=json.load(open('out/S.json'));br=np.load('out/brainmask.npy')
# --- vessels: named + schematic + population surface (probability >= 25%)
ves=np.zeros(SH,bool)
for f in ['named_vessels.pkl','schematic.pkl']:
    for o in pickle.load(open('out/'+f,'rb')):
        if o['group'] in ('Arteries','Veins & sinuses'): ves.flat[o['vox']]=True
for o in pickle.load(open('out/vasc.pkl','rb'))[0]: ves.flat[o['vox']]=True
dves=ndi.distance_transform_edt(~ves)
# --- sulcal CSF: dark voxels inside the closed brain hull that are not ventricles
hull=ndi.binary_fill_holes(ndi.binary_closing(br,iterations=5))
ven=np.isin(lab,[s['id'] for s in S if s['group']=='Ventricles'])
sul=hull&(t1<80)&~ven&~ndi.binary_erosion(hull,iterations=30)   # surface CSF only, not deep cisterns
sul=ndi.binary_opening(sul,iterations=0) if False else sul
dsul=ndi.distance_transform_edt(~sul)
print('vessel vox',ves.sum(),'sulcal vox',sul.sum())
# --- critical tract direction field (structure tensor of streamline segments), dilated 2 mm
tracks,cl,names=load_tt()
CRIT=[('CST',[62,63]),('Corticobulbar',[64,65]),('Arcuate fasciculus',[0,1]),('SLF',[28,29,30,31]),('IFOF',[20,21]),('Optic radiation',[50,51]),('Medial lemniscus',[72,73]),('Frontal aslant',[16,17]),('Uncinate',[32,33]),('ILF',[22,23])]
T=np.zeros(SH+(6,),np.float32);tid=np.zeros(SH,np.uint8)
for k,(nm,ids) in enumerate(CRIT):
    for i in np.nonzero(np.isin(cl,ids))[0]:
        p=tracks[i];d=np.diff(p,axis=0);n=np.linalg.norm(d,axis=1,keepdims=True)+1e-9;d/=n;m=(p[:-1]+p[1:])/2
        v=np.round(m-O).astype(int);ok=np.all((v>=0)&(v<SH),1);v=v[ok];d=d[ok]
        comp=np.stack([d[:,0]*d[:,0],d[:,1]*d[:,1],d[:,2]*d[:,2],d[:,0]*d[:,1],d[:,0]*d[:,2],d[:,1]*d[:,2]],1)
        np.add.at(T,(v[:,0],v[:,1],v[:,2]),comp)
        tid[v[:,0],v[:,1],v[:,2]]=np.where(tid[v[:,0],v[:,1],v[:,2]]==0,k+1,tid[v[:,0],v[:,1],v[:,2]])
has=tid>0
_,ind=ndi.distance_transform_edt(~has,return_indices=True);dist=ndi.distance_transform_edt(~has)
near=dist<=2.0
src=tuple(ind[:,near])
tidd=np.zeros(SH,np.uint8);tidd[near]=tid[src]
Tn=T[src];M=np.zeros((len(Tn),3,3),np.float32)
M[:,0,0],M[:,1,1],M[:,2,2],M[:,0,1],M[:,0,2],M[:,1,2]=Tn.T;M[:,1,0]=M[:,0,1];M[:,2,0]=M[:,0,2];M[:,2,1]=M[:,1,2]
w,vv=np.linalg.eigh(M);pd=vv[:,:,2]
dirv=np.zeros(SH+(3,),np.int8);dirv[near]=np.round(pd*120).astype(np.int8)
print('critical tract voxels',near.sum())
# --- entry candidates on the brain surface (every ~3 mm), with outward normals
sm=ndi.gaussian_filter(br.astype(np.float32),2.0);g=np.stack(np.gradient(sm),-1)
surf=br&~ndi.binary_erosion(br)
idx=np.array(np.nonzero(surf)).T
cell=(idx//3);_,first=np.unique(cell[:,0]*100000+cell[:,1]*300+cell[:,2],return_index=True);idx=idx[first]
nrm=-g[idx[:,0],idx[:,1],idx[:,2]];nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)+1e-9
mni=idx+O
keep=mni[:,2]>-35   # exclude skull-base surface for free transcortical entries
idx,nrm=idx[keep],nrm[keep]
print('candidates',len(idx))
secs=[];off=[0]
def put(a):
    b=np.ascontiguousarray(a).tobytes();n=len(b);b+=b'\0'*((-len(b))%4);o=off[0];secs.append(b);off[0]+=len(b);return [o,n]
H=dict(crit=[c[0] for c in CRIT])
H['dves']=put(np.clip(dves*8,0,255).astype(np.uint8).transpose(2,1,0))
H['dsul']=put(np.clip(dsul*8,0,255).astype(np.uint8).transpose(2,1,0))
H['ctid']=put(tidd.transpose(2,1,0))
H['cdir']=put(np.ascontiguousarray(dirv.transpose(2,1,0,3)))
H['cand']=put(idx.astype(np.int16));H['cnrm']=put(np.round(nrm*120).astype(np.int8))
hj=json.dumps(H).encode();hj+=b' '*((-len(hj))%4)
blob=np.uint32(len(hj)).tobytes()+hj+b''.join(secs)
g2=gzip.compress(blob,9);open('../docs/plan-data.wasm','wb').write(g2);print('plan data MB',len(g2)/1e6)
