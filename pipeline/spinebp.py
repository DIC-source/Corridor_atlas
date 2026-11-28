# Spine bone meshes for the Pathways view (brain to sacrum), from BodyParts3D (DBCLS, CC BY-SA 2.1 JP), a generic
# reference model of the adult body. Vertebrae C1-L5, sacrum, intervertebral discs and ribs are decimated, flipped into
# the atlas orientation (x+ = right, y+ = anterior, z+ = up), rotated and moved so the upper cervical canal continues the
# atlas cord below the foramen magnum. Per-level landmarks (canal centre and size, body, pedicles, pars, lamina,
# spinous and transverse tips, transverse foramina) are measured on the mesh and stored for labels and the schematic
# cord, roots and sympathetic trunk.  -> docs/spine-data.wasm
import numpy as np,struct,json,gzip,os,sys
import fast_simplification as fs
from scipy import ndimage as ndi
SRC=sys.argv[1] if len(sys.argv)>1 else '../data/raw/bodyparts3d/stl/'
def load(p):
    b=open(p,'rb').read()
    if b[:5]==b'solid' and b'facet' in b[:400]:
        import re;t=np.array(re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)',b),float).reshape(-1,3,3)
    else:
        n=struct.unpack('<I',b[80:84])[0];a=np.frombuffer(b[84:84+n*50],dtype=np.dtype([('n','<3f4'),('v','<9f4'),('a','<u2')]));t=a['v'].reshape(-1,3,3).astype(float)
    v,inv=np.unique(np.round(t.reshape(-1,3),4),axis=0,return_inverse=True);f=inv.reshape(-1,3)
    f=f[(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])];return v,f
VERT=[('C1',12519),('C2',12520),('C3',12521),('C4',12522),('C5',12523),('C6',12524),('C7',12525),
 ('T1',9165),('T2',9187),('T3',9209),('T4',9248),('T5',9922),('T6',9945),('T7',9968),('T8',9991),('T9',10014),('T10',10037),('T11',10059),('T12',10081),
 ('L1',13072),('L2',13073),('L3',13074),('L4',13075),('L5',13076)]
DISC=[('C2',25058),('C3',13896),('C4',13897),('C5',13898),('C6',13899),('C7',13900),('T1',10458),('T2',13495),('T3',13500),('T4',13501),('T5',13502),('T6',13503),('T7',13504),('T8',13505),('T9',13506),('T10',13507),('T11',13508),('T12',13509),
 ('L1',16033),('L2',16034),('L3',16035),('L4',16036),('L5',16037)]
RIB={1:(7857,7987),2:(7882,8012),3:(7909,8039),4:(7957,8148),5:(8066,8093),6:(8175,8202),7:(8229,8256),8:(8283,8310),9:(8364,8391),10:(8445,8472),11:(8531,8532),12:(8533,8534)}
def flip(v):return v*np.array([-1,-1,1])         # BodyParts3D: x+ = left, y+ = posterior (ribs run to -y); atlas: x+ = right, y+ = anterior. Two flips = 180° turn, so the winding stays
M={}
for n,i in VERT:M['v_'+n]=load(SRC+'FMA%d.stl'%i)
M['sacrum']=load(SRC+'FMA16202.stl')
for n,i in DISC:M['d_'+n]=load(SRC+'FMA%d.stl'%i)
for k,(r,l) in RIB.items():M['r_R%d'%k]=load(SRC+'FMA%d.stl'%r);M['r_L%d'%k]=load(SRC+'FMA%d.stl'%l)
for k in M:v,f=M[k];M[k]=(flip(v),f)
def samples(v,f,k=4):
    a,b,c=v[f[:,0]],v[f[:,1]],v[f[:,2]];r=np.random.default_rng(0).random((len(f),k,2));m=r.sum(2)>1;r[m]=1-r[m]
    return np.concatenate([v,(a[:,None]+(b-a)[:,None]*r[...,:1]+(c-a)[:,None]*r[...,1:]).reshape(-1,3)])
def holes(P,z0,z1,res=0.5):
    """project the points of a z band to xy, return occupancy grid, hole labels, grid origin"""
    q=P[(P[:,2]>=z0)&(P[:,2]<=z1)][:,:2];o=q.min(0)-6;sh=np.ceil((q.max(0)+6-o)/res).astype(int)+1;g=np.zeros(sh,bool)
    ij=np.round((q-o)/res).astype(int);g[ij[:,0],ij[:,1]]=True;g=ndi.binary_closing(ndi.binary_dilation(g,iterations=2),iterations=3)
    hl,n=ndi.label(ndi.binary_fill_holes(g)&~g);return g,hl,n,o,res
def canal_and_tf(P,zc,band=5,reg='T'):
    g,hl,n,o,res=holes(P,zc-band,zc+band);best=None;tf=[]
    if n==0:return None,[]
    for i in range(1,n+1):
        ij=np.argwhere(hl==i);xy=ij*res+o;area=len(ij)*res*res
        if area<4:continue
        c=xy.mean(0);rec=dict(c=c,area=area,lo=xy.min(0),hi=xy.max(0))
        if best is None or area>best['area']:best,old=rec,best
        tf.append(rec)
    tf=[t for t in tf if t is not best and t['area']<120]
    return best,tf
L={}
P_all={k:samples(*M[k]) for k in M if k.startswith('v_') or k=='sacrum'}
# ---- rigid placement: C1 canal centre to the atlas cord just below the foramen magnum, upper cervical canal along the atlas cord
def canal_c(k):
    P=P_all[k];z=np.median(P[:,2]);b,_=canal_and_tf(P,z,5);return np.array([b['c'][0],b['c'][1],z])
c1,c5=canal_c('v_C1'),canal_c('v_C5')
s_bp=(c1[1]-c5[1])/(c1[2]-c5[2]);s_at=(-56.2+45.6)/(-160+72)   # dy/dz of the canal in BodyParts3D vs the atlas cervical cord
ang=0.0;ca,sa=1.0,0.0   # keep the reference posture (rotating about C1 would swing the sacrum by ~10 cm)
R=np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
def place(v):return (v-c1)@R.T+np.array([0,-46.8,-82.0])
for k in M:v,f=M[k];M[k]=(place(v),f)
P_all={k:place(P) for k,P in P_all.items()}
print('cervical tilt corrected by %.1f deg'%np.degrees(ang))
# ---- per-level landmarks
LV=[n for n,_ in VERT]
for n in LV:
    P=P_all['v_'+n];reg='C' if n[0]=='C' else n[0]
    # midline slab: anterior-most solid run = body, the first gap behind it = canal (between body and lamina)
    mid=P[np.abs(P[:,0])<4];yb=np.round(mid[:,1]/0.5).astype(int);occ=np.zeros(yb.max()-yb.min()+1,bool);occ[yb-yb.min()]=True;occ=ndi.binary_closing(np.r_[False,occ,False],structure=np.ones(7))[1:-1]   # ignore gaps < 3 mm (vascular channels)
    ys=(np.arange(len(occ))+yb.min())*0.5;k=len(occ)-1
    while k>0 and not occ[k]:k-=1
    k0=k
    while k>0 and occ[k]:k-=1          # back across the body (or C1 anterior arch)
    yfront=ys[k0];yback_body=ys[k+1];k1=k
    while k>0 and not occ[k]:k-=1      # the canal gap
    ylam=ys[k];yc=(yback_body+ylam)/2;b=(yback_body-ylam)/2
    if n=='C1':yc=ylam+9;b=9.0           # dens fills the front of the C1 ring
    body=mid[mid[:,1]>=yback_body-0.5] if n!='C1' else mid[mid[:,1]>=yback_body-0.5]
    if n=='C2':body=body[body[:,2]<np.percentile(body[:,2],70)]
    top,bot=(P[:,2].max(),P[:,2].min()) if n=='C1' else (np.percentile(body[:,2],99),np.percentile(body[:,2],1))
    zmed=(top+bot)/2
    # canal width at the pedicles: x gap around the midline in a slab at the canal depth
    sl=P[(np.abs(P[:,1]-yc)<2.5)&(P[:,2]>bot)&(P[:,2]<top)];xs=np.sort(np.abs(sl[:,0]));a=float(xs[0]) if len(xs) else 10.0
    xr=sl[sl[:,0]>0][:,0];xl=-sl[sl[:,0]<0][:,0];a=float((xr.min() if len(xr) else 10)+(xl.min() if len(xl) else 10))/2;cx=0.0
    bw=P[(P[:,1]>=yback_body)&(P[:,2]>bot)&(P[:,2]<top)];rx=(np.percentile(np.abs(bw[:,0]),97)) if len(bw) else 8;ry=(yfront-yback_body)/2;bc=np.array([0,(yfront+yback_body)/2,zmed])
    _,tf=canal_and_tf(P,zmed,3,reg) if reg=='C' else (None,[])
    sp=P[np.argmin(P[:,1])]
    d=dict(n=n,top=float(top),bot=float(bot),cz=float(zmed),cx=float(cx),cy=float(yc),a=float(a),b=float(b),body=[float(bc[1]),float(rx),float(ry)],spin=sp.round(1).tolist())
    for s,key in [(1,'R'),(-1,'L')]:
        S=P[P[:,0]*s>0]
        tp=S[np.argmax(np.abs(S[:,0]))];d['tp'+key]=tp.round(1).tolist()
        ped=S[(np.abs(S[:,1]-yc)<b+2)&(np.abs(S[:,0]-cx)>a)&(np.abs(S[:,0]-cx)<a+11)]
        if len(ped):d['ped'+key]=ped.mean(0).round(1).tolist()
        par=S[(S[:,1]<yc-b+3)&(S[:,1]>yc-b-8)&(np.abs(S[:,0]-cx)>a-1)&(np.abs(S[:,0]-cx)<a+12)]
        if len(par):d['pars'+key]=par.mean(0).round(1).tolist();d['parsTop'+key]=float(par[:,2].max());d['parsBot'+key]=float(par[:,2].min())
        lam=S[(S[:,1]<yc-b)&(np.abs(S[:,0]-cx)<a*0.8)]
        if len(lam):d['lam'+key]=lam.mean(0).round(1).tolist()
        t=[t for t in (tf or []) if (t['c'][0]-cx)*s>a and t['area']<80]
        if reg=='C' and t:tt=min(t,key=lambda r:abs(abs(r['c'][0]-cx)-(a+7)));d['tf'+key]=[round(float(tt['c'][0]),1),round(float(tt['c'][1]),1),round(float(zmed),1)]
    L[n]=d
# sacrum: split into S1-S5 by height, canal per sub-level
P=P_all['sacrum'];zt,zb=np.percentile(P[:,2],99.5),P[:,2].min();fr=[.3,.24,.19,.15,.12];z=zt
for i,f in enumerate(fr):
    h=(zt-zb)*f;zc=z-h/2;n='S%d'%(i+1)
    mid=P[(np.abs(P[:,0])<4)&(np.abs(P[:,2]-zc)<h/2)];yb=np.round(mid[:,1]/0.5).astype(int);occ=np.zeros(yb.max()-yb.min()+1,bool);occ[yb-yb.min()]=True;occ=ndi.binary_closing(np.r_[False,occ,False],structure=np.ones(7))[1:-1]
    ys=(np.arange(len(occ))+yb.min())*0.5;k=len(occ)-1
    while k>0 and not occ[k]:k-=1
    while k>0 and occ[k]:k-=1
    yback=ys[k+1]
    while k>0 and not occ[k]:k-=1
    dors=float(np.percentile(mid[:,1],2));yc=dors+(10 if i<3 else 6);b=float([7,6,5,3.5,2.5][i]);a=float(max(4,12-2*i));cx=0.0   # sacral canal: a fixed depth in front of the median crest (the tilted base defeats a horizontal profile)
    band=P[np.abs(P[:,2]-zc)<h/2];body=band[band[:,1]>yc+b+1];bc=body.mean(0) if len(body) else np.array([0,yc+12,zc])
    rx=(np.percentile(body[:,0],99)-np.percentile(body[:,0],1))/2 if len(body)>20 else 8;ry=(np.percentile(body[:,1],99)-np.percentile(body[:,1],1))/2 if len(body)>20 else 6
    L[n]=dict(n=n,top=float(z),bot=float(z-h),cz=float(zc),cx=cx,cy=yc,a=a,b=b,body=[float(bc[1]),float(rx),float(ry)],ala=float(np.abs(band[:,0]).max()))
    z-=h
for n in L:print(n,'z %.0f..%.0f'%(L[n]['bot'],L[n]['top']),'canal y %.1f a %.1f b %.1f cx %.1f'%(L[n]['cy'],L[n]['a'],L[n]['b'],L[n]['cx']),'tf' if 'tfR' in L[n] else '')
# ---- decimate and pack
secs=[];off=[0];parts=[]
def put(a):
    b=np.ascontiguousarray(a).tobytes();n=len(b);b+=b'\0'*((-n)%4);o=off[0];secs.append(b);off[0]+=len(b);return [o,n]
TGT={'v':2600,'s':5000,'d':500,'r':900}
for k,(v,f) in M.items():
    kind=k[0] if k!='sacrum' else 's';t=TGT[kind]
    if len(f)>t:
        v2,f2=fs.simplify(v.astype(np.float32),f.astype(np.int32),target_reduction=1-t/len(f))
    else:v2,f2=v,f
    parts.append(dict(id=k,kind=kind,v=put(np.round(v2*10).astype(np.int16)),f=put(f2.astype(np.uint16))))
H=dict(parts=parts,L=L,src='BodyParts3D, (c) The Database Center for Life Science, licensed under CC Attribution-Share Alike 2.1 Japan')
hj=json.dumps(H,separators=(',',':')).encode();hj+=b' '*((-len(hj))%4)
blob=np.uint32(len(hj)).tobytes()+hj+b''.join(secs);g=gzip.compress(blob,9,mtime=0);open('../docs/spine-data.wasm','wb').write(g)
print('parts',len(parts),'raw %.2f MB gz %.2f MB'%(len(blob)/1e6,len(g)/1e6))
