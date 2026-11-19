import json,numpy as np,pickle,gzip
OUT='out/'
lab=np.load(OUT+'lab.npy'); t1=np.load(OUT+'t1.npy'); t2=np.load(OUT+'t2.npy'); S=json.load(open(OUT+'S.json'))
MESH,TR,T=pickle.load(open(OUT+'geo.pkl','rb'))
secs=[];off=[0]
def orient(v,f):
    a,b,c=v[f[:,0]],v[f[:,1]],v[f[:,2]]
    vol=np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6
    return f if vol>0 else f[:,[0,2,1]]
nflip=[0]
def put(a):
    b=np.ascontiguousarray(a).tobytes(); b+=b'\0'*((-len(b))%4); o=off[0]; secs.append(b); off[0]+=len(b); return [o,len(np.ascontiguousarray(a).tobytes())]
# store volumes in C order of (z,y,x) so x is fastest
H=dict(dims=[181,217,181],origin=[-90,-126,-72])
H['t1']=put((t1//4*4).astype(np.uint8).transpose(2,1,0)); H['t2']=put((t2//4*4).astype(np.uint8).transpose(2,1,0)); H['lab']=put(lab.transpose(2,1,0))
for s,m in zip(S,MESH):
    if m is None: continue
    v,f=m; f=orient(v,f); s['v']=put(np.round(v*20).astype(np.int16)); s['f']=put(f.astype(np.uint16 if len(v)<65536 else np.uint32)); s['f32']=len(v)>=65536
H['S']=S
tr=[]
sh=(181,217,181)
for t in TR:
    # re-index voxels to z,y,x order
    i,j,k=np.unravel_index(t['vox'],sh); nv=np.sort((k*217+j)*181+i).astype(np.uint32)
    d=np.diff(np.r_[0,nv]).astype(np.uint32)
    L=[put(l) for l in t['lines']]
    tr.append(dict(id=t['id'],name=t['name'],group=t['group'],color=t['color'],side=t['side'],n=t['n'],vox=put(d),lines=L))
VS,TE,terv=pickle.load(open(OUT+'vasc.pkl','rb'))
from scipy import ndimage as ndi
Vm=np.load(OUT+'vmask.npy'); dt=ndi.distance_transform_edt(Vm)*0.5; dtm=ndi.maximum_filter(dt,3)
nd,ni=ndi.distance_transform_edt(lab==0,return_indices=True)
for t in VS:
    i,j,k=np.unravel_index(t['vox'],sh); nv=np.sort((k*217+j)*181+i).astype(np.uint32)
    v,f=t['mesh']; f=f[:,[0,2,1]]
    k2,r2=np.divmod(nv,217*181); j2,i2=np.divmod(r2,181)
    ii,jj,kk=2*i2+16,2*j2+16,2*k2
    ok=(ii<Vm.shape[0])&(jj<Vm.shape[1])&(kk<Vm.shape[2])
    rad=np.zeros(len(nv)); rad[ok]=dtm[ii[ok],jj[ok],kk[ok]]
    nb=lab[ni[0][i2,j2,k2],ni[1][i2,j2,k2],ni[2][i2,j2,k2]].astype(np.uint8)
    nbd=np.minimum(nd[i2,j2,k2],25).astype(np.uint8)
    tr.append(dict(id=t['id'],name=t['name'],group=t['group'],color=t['color'],side=t['side'],n=0,vox=put(np.diff(np.r_[0,nv]).astype(np.uint32)),lines=[],v=put(np.round(v*20).astype(np.int16)),f=put(f.astype(np.uint16) if len(v)<65536 else f.astype(np.uint32)),f32=len(v)>=65536,rad=put(np.round(rad*10).astype(np.uint8)),nb=put(nb),nbd=put(nbd)))
from scipy import ndimage as _ndi
import grid as _g
_fs=_g.res('FreeSurferSeg.nii.gz')>0
_br=_ndi.binary_fill_holes(_ndi.binary_closing(_fs|(lab>0),iterations=2)).astype(np.uint8)
H['brain']=put(_br.transpose(2,1,0))
H['mra']=put(np.load(OUT+'mra1.npy').transpose(2,1,0))
_m=np.load(OUT+'mrah_crop.npy');_lo=np.load(OUT+'mrah_lo.npy');_v=_m[_m>0]
_thr=np.percentile(_v,88);_top=np.percentile(_v,99.95)
_u=(np.clip((_m-_thr)/(_top-_thr),0,1)**0.8*255).astype(np.uint8)
H['mrah']=put(_u.transpose(2,1,0));H['mrahd']=list(map(int,_u.shape));H['mraho']=list(map(float,_lo*0.5+np.array([-98,-134,-72])))
te=[]
for t in TE:
    v,f=t['mesh']; f=orient(v,f); te.append(dict(id=t['id'],name=t['name'],group=t['group'],color=t['color'],side=t['side'],v=put(np.round(v*20).astype(np.int16)),f=put(f.astype(np.uint16))))
H['TE']=[]
H['T']=tr
hj=json.dumps(H,separators=(',',':')).encode()
hj+=b' '*((-len(hj))%4)
blob=np.uint32(len(hj)).tobytes()+hj+b''.join(secs)
g=gzip.compress(blob,9)
open('../site/atlas-data.bin','wb').write(g)
print(len(blob)/1e6,len(g)/1e6)
