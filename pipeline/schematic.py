# Schematic (textbook-course) surgical vessels and cranial nerves, anchored to template landmarks.
import numpy as np,json,pickle
from scipy.interpolate import splprep,splev
O=np.array([-90,-126,-72]);SH=(181,217,181);br=np.load('out/brainmask.npy')
A=json.load(open('out/arteries.json'));V=json.load(open('out/veins.json'))
src=open('build_named.py').read();ns={}
exec(src.split("AC={")[0].split("A=json.load")[0],ns)              # imports
exec("def smooth"+src.split("def smooth")[1].split("AC={")[0],ns)   # smooth, tube, voxels
smooth,tube,voxels=ns['smooth'],ns['tube'],ns['voxels']
def inside(p):
    v=np.round(np.array(p)-O).astype(int);return np.all(v>=0) and np.all(v<SH) and br[tuple(v)]
def snap(p,c,off=1.5):
    p=np.array(p,float);c=np.array(c,float);d=p-c;d/=np.linalg.norm(d);last=None;t=0
    while t<150:
        if inside(c+d*t):last=t
        t+=0.5
    return c+d*((last or 0)+off)
near=lambda key,ax,val:np.array(min(((V if key in V else A)[key]['pts']),key=lambda q:abs(q[ax]-val)))
def onpath(key,frac):p=np.array(A[key]['pts']);return p[int(frac*(len(p)-1))]

LABV=np.load('out/lab.npy');SJ=json.load(open('out/S.json'))
BS=np.isin(LABV,[q['id'] for q in SJ if q['name'] in ('Midbrain','Pons','Medulla','Cerebral peduncle (crus cerebri)')])
def bsurf(z,th,side,off):
    """point on the brainstem surface at level z; th = angle from the anterior midline (0) toward lateral (90) and posterior (180)"""
    zi=int(round(z-O[2]));zi=max(0,min(180,zi));col=BS[88:93,:,zi].any(0);ys=np.nonzero(col)[0]
    if not len(ys):
        sl=BS[:,:,zi];ys=np.nonzero(sl.any(0))[0]
    yc=(ys.min()+ys.max())/2+O[1];d=np.array([side*np.sin(np.radians(th)),np.cos(np.radians(th)),0.]);c=np.array([0,yc,z],float);last=0
    for t in np.arange(0,40,0.25):
        v=np.round(c+d*t-O).astype(int)
        if np.all(v>=0) and np.all(v<SH) and BS[tuple(v)]:last=t
    return (c+d*(last+off)).tolist()
OUT=[]
def put(id_,name,group,color,side,pts,r,note,kind):
    P=smooth(pts,s=0) if len(pts)<5 else smooth(pts,s=len(pts)*0.3)
    rr=np.interp(np.linspace(0,1,len(P)),np.linspace(0,1,len(r)),r) if np.ndim(r) else np.full(len(P),r)
    mv,mf=tube(P,rr,seg=10);vox,rv=voxels(P,np.maximum(rr,0.6))
    OUT.append(dict(id=id_,name=name,group=group,color=color,side=side,vox=vox,radv=rv,mesh=(mv,mf),sch=True,note=note,kind=kind))
mir=lambda pts,m:[[p[0]*m,p[1],p[2]] for p in pts]
AR,VE,NV='#e06a5c','#5b8fd8','#e2b928'
for sd,m in [('L',1),('R',-1)]:
    s=sd.lower();k=lambda n:n+'_'+s
    # ---- cranial nerves (left coordinates, x mirrored for right)
    sd_=-1  # left side in MNI = negative x; mir() flips for right
    L_=lambda z,th,off=1.0:[*bsurf(z,th,-1,off)]
    CN={'cn1':('Olfactory nerve / tract (CN I)',[[-7,44,-21],[-8,32,-20],[-11,20,-19],[-16,9,-17]],[2.2,1.2,1.0,1.0]),
        'cn3':('Oculomotor nerve (CN III)',[[-4,-14,-14],[-8,-9,-17],[-12,-4,-21],[-14,0,-25]],1.1),
        'cn4':('Trochlear nerve (CN IV)',[L_(-17,178,0.5),L_(-18,135,1),L_(-19,90,1.5),[-17,-14,-23],[-15,-5,-27]],0.5),
        'cn5':('Trigeminal nerve (CN V)',[L_(-31,62,0),L_(-31,62,4),[-25,-15,-30],[-28,-10,-32]],[2.4,2.3,2.1,2.0]),
        'cn6':('Abducens nerve (CN VI)',[L_(-48,12,0),L_(-44,10,2.5),[-8,-11,-38],[-11,-7,-32],[-14,-3,-28]],0.7),
        'cn7':('Facial nerve (CN VII)',[L_(-45,78,0),L_(-44,80,4),[-28,-26,-41],[-34,-24,-38]],0.9),
        'cn8':('Vestibulocochlear nerve (CN VIII)',[L_(-46,92,0),L_(-45,92,4),[-28,-29,-43],[-34,-28,-41]],1.3),
        'cn9':('Glossopharyngeal nerve (CN IX)',[L_(-52,112,0),L_(-52,108,4),[-24,-33,-55],[-31,-31,-57]],0.8),
        'cn10':('Vagus nerve (CN X)',[L_(-55,115,0),L_(-55,110,4),[-24,-34,-57],[-31,-32,-59]],1.0),
        'cn11':('Accessory nerve (CN XI)',[[-9,-45,-86],[-10,-44,-76],L_(-66,112,2),L_(-59,112,3),[-24,-35,-59],[-31,-33,-60]],0.8),
        'cn12':('Hypoglossal nerve (CN XII)',[L_(-61,38,0),L_(-62,40,3),[-14,-34,-66],[-19,-33,-70]],0.9)}
    for key,(nm,pts,r) in CN.items():put('cs_'+k(key),nm,'Cranial nerves',NV,sd,mir(pts,m),r,'schematic course','nerve')
    # ---- arteries
    a2=np.array(A['a2']['pts']);put('sa_'+k('heubner'),'Recurrent artery of Heubner','Arteries',AR,sd,mir([[-1,19,-9],[-6,16,-9],[-12,10,-11],[-17,6,-11],[-16,8,-5],[-14,10,2]],m),0.55,'schematic course','artery')
    m1=np.array(A[k('m1')]['pts'])
    for j,x in enumerate([-20,-26,-32]):
        q=m1[np.argmin(np.abs(m1[:,0]-x*m))]
        put('sa_'+k('lsa%d'%(j+1)),'Lateral lenticulostriate artery','Arteries',AR,sd,[q.tolist(),[q[0]*0.95,q[1]-1,-6],[q[0]*0.86,q[1]-3,6],[q[0]*0.82,q[1]-4,13]],0.45,'schematic course','artery')
    ic=np.array(A[k('ica')]['pts'])[-6]
    put('sa_'+k('acha'),'Anterior choroidal artery','Arteries',AR,sd,[ic.tolist()]+mir([[-17,-6,-18],[-21,-14,-14],[-25,-22,-10],[-28,-28,-6]],m),0.6,'schematic course','artery')
    g=int(np.argmax(a2[:,1]));c0=np.array([0,-5,10.0]);cm=[]
    for q in a2[g:]:
        if q[1]<-25:break
        d=q-c0;d[0]=0;d/=np.linalg.norm(d);p=q+d*11;p[0]=-3*m;cm.append(p.tolist())
    put('sa_'+k('cma'),'Callosomarginal artery','Arteries',AR,sd,cm[::3],0.8,'schematic course','artery')
    bq=near('ba',2,-43)
    put('sa_'+k('aica'),'Anterior inferior cerebellar artery (AICA)','Arteries',AR,sd,[bq.tolist()]+mir([L_(-43,35,2.2),L_(-43,75,3),[-24,-29,-44],[-29,-27,-42],[-26,-33,-45],[-31,-42,-48]],m),0.8,'schematic course: from the lower basilar, loops at the CN VII/VIII complex','artery')
    va=np.array(A[k('va')]['pts']);vq=va[np.argmin(np.abs(va[:,2]+61))]
    put('sa_'+k('pica'),'Posterior inferior cerebellar artery (PICA)','Arteries',AR,sd,[vq.tolist()]+mir([L_(-62,80,2.2),L_(-63,125,2.5),L_(-66,160,3),[-6,-55,-66],[-5,-60,-58],[-8,-66,-50]],m),0.8,'schematic course: lateral medullary, tonsillomedullary and telovelotonsillar segments','artery')
    tip=np.array(A['ba']['pts'][-1])
    put('sa_'+k('sca2'),'Superior cerebellar artery (SCA)','Arteries',AR,sd,[(tip+[0,0,-1.5]).tolist()]+mir([L_(-21,30,2),L_(-22,80,2.2),L_(-21,130,2.5),L_(-20,170,3),[-8,-58,-18],[-14,-64,-20]],m),1.0,'schematic course around the pontomesencephalic junction','artery')
    # ---- veins
    sylv=[snap([-50*m,y,z],[0,y,10]) for y,z in [(-28,16),(-16,8),(-4,0),(8,-8),(14,-14)]]
    put('sv_'+k('smcv'),'Superficial middle cerebral (Sylvian) vein','Veins & sinuses',VE,sd,[p.tolist() for p in sylv]+mir([[-32,16,-20],[-24,10,-26],[-17,2,-28]],m),[1.4,1.6,1.8,1.8],'schematic course','vein')
    tsj=np.array(V['ts_'+s]['pts'][-1])
    lab_=[sylv[1]]+[snap([-50*m,y,z],[0,y,0]) for y,z in [(-30,-8),(-46,-14),(-60,-18)]]+[tsj]
    put('sv_'+k('labbe'),'Vein of Labbé','Veins & sinuses',VE,sd,[p.tolist() for p in lab_],1.4,'schematic course; very variable','vein')
    sssq=near('sss',1,-22)
    tro=[sylv[1]]+[snap([-40*m,y,z],[0,y,10]) for y,z in [(-18,30),(-20,48),(-22,62)]]+[sssq]
    put('sv_'+k('trolard'),'Vein of Trolard','Veins & sinuses',VE,sd,[p.tolist() for p in tro],1.3,'schematic course; very variable','vein')
    for j,y in enumerate([25,0,-30]):
        sq=near('sss',1,y);p0=snap([-22*m,y,sq[2]-12],[0,y,10])
        put('sv_'+k('bridge%d'%(j+1)),'Bridging vein to SSS','Veins & sinuses',VE,sd,[p0.tolist(),[(p0[0]+sq[0])/2,y,(p0[2]+sq[2])/2+1.5],sq.tolist()],1.0,'schematic; number and position vary','vein')
    put('sv_'+k('tsv'),'Thalamostriate vein','Veins & sinuses',VE,sd,mir([[-20,-22,17],[-16,-12,15],[-10,-3,13],[-4,2,10]],m),1.0,'schematic course (caudothalamic groove)','vein')
    put('sv_'+k('septal'),'Septal vein','Veins & sinuses',VE,sd,mir([[-3,24,14],[-3,14,13],[-3,4,11],[-3.5,2,10]],m),0.8,'schematic course','vein')
    put('sv_'+k('dandy'),'Superior petrosal vein (Dandy)','Veins & sinuses',VE,sd,mir([[-22,-36,-30],[-27,-33,-28],[-31,-30,-27]],m),1.2,'schematic course','vein')
    put('sv_'+k('sps'),'Superior petrosal sinus','Veins & sinuses',VE,sd,mir([[-17,-4,-30],[-26,-18,-28],[-33,-30,-27],[-42,-50,-24]],m)+[tsj.tolist()],1.5,'schematic course (petrous ridge)','vein')
    put('sv_'+k('cav'),'Cavernous sinus','Veins & sinuses',VE,sd,mir([[-15,-12,-30],[-16,-4,-29],[-16,6,-27]],m),3.0,'schematic position','vein')
torc=np.array(V['sss']['pts'][-1])
occ=[torc]+[snap([0,y,z],[0,-55,-30],2) for y,z in [(-95,-25),(-82,-42),(-64,-58)]]
put('sv_occ','Occipital sinus','Veins & sinuses',VE,'',[p.tolist() for p in occ],1.4,'schematic course; often small or absent','vein')
for o in OUT:print(o['id'],o['name'],len(o['vox']))
pickle.dump(OUT,open('out/schematic.pkl','wb'))
