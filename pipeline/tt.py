import gzip,scipy.io as sio,io,numpy as np
D='../data/raw/'
def load_tt():
    m=sio.loadmat(io.BytesIO(gzip.decompress(open(D+'tract/human.tt.gz','rb').read())))
    buf=m['track'].ravel().tobytes(); T=m['trans_to_mni'].ravel().reshape(4,4)
    tracks=[];i=0;n=len(buf)
    while i<n:
        c=int.from_bytes(buf[i:i+4],'little'); i+=4
        x,y,z=np.frombuffer(buf[i:i+12],'<i4'); i+=12
        k=(c-3)//3
        d=np.frombuffer(buf[i:i+k*3],np.int8).reshape(-1,3).astype(np.int64); i+=k*3
        p=np.vstack([[x,y,z],d]).cumsum(0)/32.0
        tracks.append(p)
    cl=m['cluster'].ravel()
    names=[l.strip() for l in open(D+'tract/human.txt') if l.strip()]
    # voxel->mni
    out=[]
    for p in tracks:
        h=np.c_[p,np.ones(len(p))]@T.T
        out.append(h[:,:3])
    return out,cl,names
if __name__=='__main__':
    t,cl,names=load_tt()
    print(len(t),len(cl),len(names),cl.max())
    import collections
    c=collections.Counter(cl.tolist())
    for k in sorted(c)[:200]: print(k,names[k] if k<len(names) else '?',c[k])
    a=np.vstack(t); print(a.min(0),a.max(0))
