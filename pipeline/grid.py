import nibabel as nib, numpy as np
from scipy.ndimage import map_coordinates
D='../data/raw/dsa/human/'
O=np.array([-90,-126,-72.]); S=(181,217,181)
X,Y,Z=np.meshgrid(*[O[i]+np.arange(S[i]) for i in range(3)],indexing='ij')
W=np.stack([X,Y,Z,np.ones_like(X)],0).reshape(4,-1)
def res(fn,order=0,shift=(0,0,0),vol=None):
    im=nib.load(D+fn); a=im.affine; d=np.asanyarray(im.dataobj)
    if vol is not None: d=d[...,vol]
    w=W.copy(); w[:3]-=np.array(shift)[:,None]
    v=np.linalg.inv(a)@w
    return map_coordinates(d.astype(np.float32),v[:3],order=order,mode='constant').reshape(S)
