# Sulcal / fissural landmark lines from the cortical parcellation (left labels 1-31, right = +31).
import numpy as np
from scipy import ndimage as ndi
O=np.array([-90,-126,-72]);SH=(181,217,181)
LAB=np.load('out/lab.npy');BR=np.load('out/brainmask.npy')>0
_hull=ndi.binary_fill_holes(ndi.binary_closing(BR|(LAB>0),structure=ndi.generate_binary_structure(3,1),iterations=4))
DEPTH=ndi.distance_transform_edt(_hull)          # mm below the outer (hull) surface
_ctx=(LAB>=1)&(LAB<=62)
_,_ni=ndi.distance_transform_edt(~_ctx,return_indices=True)
NL=np.where(_hull,LAB[_ni[0],_ni[1],_ni[2]],0)*(_hull)   # every hull voxel gets its nearest cortical label (fills parcellation gaps)
NL=np.where((NL>=1)&(NL<=62),NL,0)
_cache={}
def lid(i,side):return i if side=='l' else i+31
def adj(A,B,side,maxd=4.0,mind=0.0):
    """voxels of label set A touching label set B, within [mind,maxd] mm of the outer hull; returns MNI coords"""
    key=(tuple(A),tuple(B),side,maxd,mind)
    if key in _cache:return _cache[key]
    a=np.isin(NL,[lid(i,side) for i in A]);b=np.isin(NL,[lid(i,side) for i in B])
    nb=ndi.binary_dilation(b,structure=np.ones((3,3,3)))
    m=a&nb&(DEPTH<=maxd)&(DEPTH>=mind)
    w=np.argwhere(m)+O;_cache[key]=w;return w
def line(w,axis,lo,hi,step=5.0,minn=3):
    """bin points along an axis and return the per-bin median (ordered lo->hi)"""
    out=[]
    for c in np.arange(lo,hi+1e-6,step if hi>=lo else -step):
        s=w[np.abs(w[:,axis]-c)<=step/2]
        if len(s)>=minn:p=np.median(s,0);p[axis]=c;out.append(p)
    return np.array(out)
def surf(p,d=1.0):
    """push a point outward (up the depth gradient) until it sits d mm under the hull surface"""
    p=np.array(p,float)
    g=np.array(np.gradient(DEPTH))
    for _ in range(40):
        v=np.round(p-O).astype(int)
        if np.any(v<0) or np.any(v>=SH):break
        dd=DEPTH[tuple(v)]
        if dd<=d:break
        gr=g[:,v[0],v[1],v[2]];n=np.linalg.norm(gr)
        if n<1e-6:break
        p=p-gr/n*0.5
    return p
