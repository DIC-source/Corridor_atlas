# PAM50 spinal cord template (De Leener et al., NeuroImage 2018) -> docs/spine-mri.wasm
# T2 (uint8, 0.5 mm in-plane, 1 mm slices), a label volume (36 white/grey-matter atlas regions by maximum probability,
# CSF = 40), the centreline and the vertebral-level ranges along the template's straightened z axis.
import nibabel as nib,numpy as np,gzip,json,sys
P=sys.argv[1] if len(sys.argv)>1 else '../data/raw/pam50/'
ld=lambda f:np.asarray(nib.load(P+f).dataobj)
t2=ld('template/PAM50_t2.nii.gz').astype(np.float32)
lo,hi=np.percentile(t2[t2>0],[0.5,99.7]);T2=np.clip((t2-lo)/(hi-lo)*255,0,255).astype(np.uint8)
A=np.stack([ld('atlas/PAM50_atlas_%02d.nii.gz'%i).astype(np.float32) for i in range(36)],0)
mx=A.max(0);lab=np.where(mx>0.25,A.argmax(0)+1,0).astype(np.uint8)
csf=ld('template/PAM50_csf.nii.gz')>0.5;lab[(lab==0)&csf]=40
lev=ld('template/PAM50_levels.nii.gz');cl=ld('template/PAM50_centerline.nii.gz')
K=T2.shape[2];cen=[]
for k in range(K):
    w=np.argwhere(cl[:,:,k]>0);cen.append([float(w[:,0].mean()),float(w[:,1].mean())] if len(w) else None)
last=None
for k in range(K):
    if cen[k] is None:cen[k]=last
    else:last=cen[k]
first=next(c for c in cen if c);cen=[c or first for c in cen]
levels={}
for v in range(1,21):
    z=np.nonzero((lev==v).any((0,1)))[0]
    if len(z):levels[v]=[int(z.min()),int(z.max())]
sl=slice(None,None,2)   # 1 mm slices
secs=[];off=[0]
def put(a):
    b=np.ascontiguousarray(a).tobytes();n=len(b);b+=b'\0'*((-n)%4);o=off[0];secs.append(b);off[0]+=len(b);return [o,n]
# store as (k, j, i) so i is fastest
H=dict(dims=[T2.shape[0],T2.shape[1],len(range(K)[sl])],kstep=2,res=0.5,x0=35.0,y0=-81.22,
  t2=put(T2[:,:,sl].transpose(2,1,0)),lab=put(lab[:,:,sl].transpose(2,1,0)),cen=[cen[k] for k in range(K)][::2],levels=levels,
  src='PAM50 template, De Leener B et al. NeuroImage 2018;165:170-179 (Spinal Cord Toolbox)')
hj=json.dumps(H,separators=(',',':')).encode();hj+=b' '*((-len(hj))%4)
blob=np.uint32(len(hj)).tobytes()+hj+b''.join(secs);g=gzip.compress(blob,9);open('../docs/spine-mri.wasm','wb').write(g)
print('levels',levels);print('raw %.1f MB gz %.1f MB'%(len(blob)/1e6,len(g)/1e6))
