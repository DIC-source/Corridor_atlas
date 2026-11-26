# Optic radiation loops (textbook population course): Meyer loop (ventral, temporal) and Baum loop (dorsal, parietal).
# The HCP-1065 optic-radiation bundle barely reaches anterior to the LGN (group averaging truncates Meyer's loop),
# so both loops are drawn as schematic bundles from template landmarks and appended to geo.pkl as their own layers.
#   Meyer loop: LGN -> forward over the roof of the temporal horn, tip at the level of the horn tip
#               (Yogarajah et al., Brain 2009: tip 0 mm from the horn tip on average, range -15..+9; 24-47 mm behind the pole)
#               -> back along the lateral wall of the temporal horn and atrium -> lower bank of the calcarine (upper visual field)
#   Baum loop:  LGN -> up through the retrolenticular internal capsule -> parietal white matter lateral to the upper atrium
#               -> upper bank of the calcarine (lower visual field)
# Run after build2.py; idempotent (replaces earlier loop layers).
import numpy as np,json,pickle
from scipy import ndimage as ndi
from scipy.interpolate import splprep,splev
O=np.array([-90,-126,-72]);OUT='out/'
lab=np.load(OUT+'lab.npy');S=json.load(open(OUT+'S.json'));shape=lab.shape
ids={s['name']+'|'+s['side']:s['id'] for s in S}
lv=np.argwhere(lab==ids['Lateral ventricle|'])+O
rng=np.random.default_rng(7)
# the loops run in white matter: around the temporal horn and atrium, never through them, the hippocampus or the amygdala
_avoid=np.isin(lab,[ids['Lateral ventricle|']]+[ids[k] for k in ids if k.split('|')[0] in ('Hippocampus','Amygdala')])
_avoid=ndi.binary_dilation(_avoid,iterations=1)
_,_near=ndi.distance_transform_edt(_avoid,return_indices=True)
def outside(p):
    p=p.copy()
    for i in range(len(p)):
        v=np.round(p[i]-O).astype(int)
        if np.all((v>=0)&(v<np.array(shape))) and _avoid[tuple(v)]:
            q=np.array([_near[k][tuple(v)] for k in range(3)])+O;d=q-p[i];n=np.linalg.norm(d)+1e-6;p[i]=q+d/n*1.2
    from scipy.ndimage import gaussian_filter1d as g1
    return np.c_[[g1(p[:,k],1.2,mode='nearest') for k in range(3)]].T
def bundle(W,n=24,sig=1.6):
    W=np.array(W,float);out=[]
    tck,_=splprep(W.T,s=0,k=3);u=np.linspace(0,1,400);base=np.array(splev(u,tck)).T
    L=np.r_[0,np.cumsum(np.linalg.norm(np.diff(base,axis=0),axis=1))];m=max(2,int(L[-1]));base=np.array([np.interp(np.linspace(0,L[-1],m),L,base[:,k]) for k in range(3)]).T
    t=np.linspace(0,1,m);env=0.3+np.sin(np.pi*t)    # fibres converge at the LGN and spread over the calcarine bank
    for i in range(n):
        off=rng.normal(0,sig,3);end=rng.normal(0,1.8,3)*np.array([0.6,1.0,1.0])
        out.append(outside(outside(base+np.outer(env,off)+np.outer(t**3,end))))
    return out
def landmarks(sd):
    m=-1 if sd=='L' else 1
    lgn=np.argwhere(lab==ids['Lateral geniculate (LGN)|'+sd]).mean(0)+O
    v=lv[lv[:,0]*m>0];th=v[(v[:,2]<-5)&(v[:,1]>-45)];tip=th[np.argmax(th[:,1])].astype(float)
    roof=th[np.abs(th[:,1]-tip[1])<4][:,2].max()                       # top of the horn near its tip
    pc=np.argwhere(lab==ids['Pericalcarine (V1, BA17)|'+sd])+O;q=pc[np.abs(pc[:,1]+77)<3];zmid=np.median(q[:,2]);xm=np.median(q[:,0])
    return m,lgn,tip,roof,np.array([xm,-77,q[q[:,2]<zmid][:,2].mean()]),np.array([xm,-75,q[q[:,2]>zmid][:,2].mean()])
geo=pickle.load(open(OUT+'geo.pkl','rb'));MESH,TR,T=geo
TR=[t for t in TR if not t['id'].startswith(('or_mey','or_baum'))]
for t in TR:
    if t['id'] in ('or_l','or_r'):t['name']='Optic radiation (geniculocalcarine)'
LOOPS=[]
for sd in ('L','R'):
    m,lgn,tip,roof,low,up=landmarks(sd)
    X=lambda x:m*abs(x)
    mey=[lgn,[X(28),lgn[1]+8,lgn[2]-7],[X(33),tip[1]-2,roof+1],[X(39),tip[1]-12,roof-1],[X(40),-28,-11],[X(39),-42,-4],[X(33),-60,-2],[X(22),-72,low[2]],low]
    bau=[lgn,[X(29),-30,lgn[2]+8],[X(36),-40,14],[X(36),-52,17],[X(28),-66,17],[X(18),-73,up[2]+1],up]
    for key,name,col,W,vf in [('or_mey','Optic radiation · Meyer loop','#0e9aa7',mey,'upper'),('or_baum','Optic radiation · Baum loop','#5a4fcf',bau,'lower')]:
        lines=bundle(W)
        msk=np.zeros(shape,bool)
        for p in lines:
            q=np.vstack([p[:-1]+(p[1:]-p[:-1])*a for a in (0,.5)]+[p[-1:]]);v=np.round(q-O).astype(int);ok=np.all((v>=0)&(v<np.array(shape)),1);v=v[ok];msk[v[:,0],v[:,1],v[:,2]]=True
        msk=ndi.binary_closing(msk,iterations=1)|msk
        tid=key+'_'+sd.lower()
        TR.append(dict(id=tid,name=name,group='Visual',color=col,side=sd,n=len(lines),vox=np.nonzero(msk.ravel())[0].astype(np.uint32),
                       lines=[np.round((p[::max(1,len(p)//60)])*4).astype(np.int16) for p in lines]))
        LOOPS.append(dict(id=tid,lines=lines))
        print(tid,'voxels',msk.sum(),'tip y' if key=='or_mey' else 'top z',round(max(p[:,1].max() for p in lines),1) if key=='or_mey' else round(max(p[:,2].max() for p in lines),1),'| horn tip',tip,'roof',roof)
pickle.dump((MESH,TR,T),open(OUT+'geo.pkl','wb'));pickle.dump(LOOPS,open(OUT+'oloops.pkl','wb'))
