# Named arteries (traced on the population probability atlas) and veins (template geometry) -> tube meshes + 1 mm voxel lists
import numpy as np,json,pickle
from scipy.interpolate import splprep,splev
import nibabel as nib
O=np.array([-90,-126,-72]);SH=(181,217,181)
A=json.load(open('out/arteries.json'));V=json.load(open('out/veins.json'))
R=nib.load('../data/raw/mouches/vesselRadius.nii').get_fdata(dtype=np.float32)[::-1];HO=np.array([-91.25,-126.25,-72.25])
from scipy import ndimage as ndi
Rm=ndi.maximum_filter(R,3)
def smooth(p,step=0.7,s=None):
    p=np.array(p,float);k=min(3,len(p)-1);tck,u=splprep(p.T,s=(len(p)*0.35 if s is None else s),k=k)
    L=np.linalg.norm(np.diff(p,axis=0),axis=1).sum();return np.array(splev(np.linspace(0,1,max(6,int(L/step))),tck)).T
def tube(P,r,seg=12):
    n=len(P);T=np.gradient(P,axis=0);T/=np.linalg.norm(T,axis=1)[:,None]+1e-9
    a=np.array([0,0,1.0]) if abs(T[0,2])<0.9 else np.array([1.0,0,0]);N=np.cross(T[0],a);N/=np.linalg.norm(N);verts=[];faces=[]
    for i in range(n):
        N=N-T[i]*np.dot(N,T[i]);N/=np.linalg.norm(N)+1e-9;B=np.cross(T[i],N)
        for k in range(seg):
            an=2*np.pi*k/seg;verts.append(P[i]+r[i]*(np.cos(an)*N+np.sin(an)*B))
    for i in range(n-1):
        for k in range(seg):
            a0=i*seg+k;a1=i*seg+(k+1)%seg;b0=a0+seg;b1=a1+seg;faces+= [[a0,a1,b0],[a1,b1,b0]]
    c0=len(verts);verts.append(P[0]);c1=len(verts);verts.append(P[-1])
    for k in range(seg):faces+=[[c0,(k+1)%seg,k],[c1,(n-1)*seg+k,(n-1)*seg+(k+1)%seg]]
    v=np.array(verts,np.float32);f=np.array(faces,np.int32)
    a_,b_,c_=v[f[:,0]],v[f[:,1]],v[f[:,2]]
    if np.einsum('ij,ij->i',a_,np.cross(b_,c_)).sum()<0:f=f[:,[0,2,1]]
    return v,f
def voxels(P,r):
    lo=np.floor(P.min(0)-r.max()-1).astype(int);hi=np.ceil(P.max(0)+r.max()+1).astype(int)
    g=np.stack(np.meshgrid(*[np.arange(lo[i],hi[i]+1) for i in range(3)],indexing='ij'),-1).reshape(-1,3).astype(float)
    best=np.full(len(g),1e9);rr=np.zeros(len(g))
    for p,ri in zip(P[::2],r[::2]):
        d=np.linalg.norm(g-p,axis=1)-ri;m=d<best;best[m]=d[m];rr[m]=ri
    q=g[best<=0.45].astype(int)-O;ok=np.all((q>=0)&(q<SH),1);q=q[ok];rr=rr[best<=0.45][ok]
    return np.ravel_multi_index(q.T,SH).astype(np.uint32),rr
AC={'ica':'#c0392b','m1':'#e2463a','m2':'#ee6a50','a1':'#e8742f','a2':'#f08b3c','pcom':'#d94b6e','p1p2':'#c43c7a','p3':'#b5487f','sca':'#a8493a','ba':'#9b2d20','va':'#8a3324'}
OUT=[]
for k,v in A.items():
    base=k.split('_')[0]
    if base=='sca': continue   # replaced by the schematic SCA anchored to the brainstem
    P=smooth(v['pts'])
    idx=np.clip(((P-HO)/0.5).round().astype(int),0,np.array(R.shape)-1);r=np.clip(Rm[tuple(idx.T)],0.7,3.0)
    r=ndi.uniform_filter1d(r,9)
    if base=='va': r=np.full(len(P),1.8)
    mv,mf=tube(P,r);vox,rv=voxels(P,r)
    OUT.append(dict(id='a_'+k,name=v['name'],group='Arteries',color=AC[base],side=v['side'],vox=vox,radv=rv,mesh=(mv,mf),**({'sch':True,'note':'schematic course: V2 in the transverse foramina, V3 over C1, V4 along the medulla'} if base=='va' else {})))
    print(k,len(P),'r %.1f-%.1f'%(r.min(),r.max()),len(vox))
for k,v in V.items():
    P=smooth(v['pts'],s=len(v['pts'])*0.6);t=np.linspace(0,1,len(P));r=v['r'][0]+(v['r'][1]-v['r'][0])*t
    mv,mf=tube(P,r);vox,rv=voxels(P,r)
    OUT.append(dict(id='vn_'+k,name=v['name'],group='Veins & sinuses',color='#3566b8' if 'sinus' in v['name'] else '#4b86d4',side=v['side'],vox=vox,radv=rv,mesh=(mv,mf)))
    print(k,len(P),len(vox))
pickle.dump(OUT,open('out/named_vessels.pkl','wb'))
