import numpy as np,json
from tt import load_tt
from skimage.measure import find_contours
from scipy.ndimage import gaussian_filter1d
tracks,cl,names=load_tt()
lab=np.load('out/lab.npy');S=json.load(open('out/S.json'));O=np.array([-90,-126,-72])
ids={s['name']+'|'+s['side']:s['id'] for s in S}
bsid=[ids['Midbrain|'],ids['Pons|'],ids['Medulla|']]
R=json.load(open('out/paths_raw.json'))
ax=np.array(R['axis']);
k=np.array([0.12,ax[0,1]+0.12*72])
def A(z):
    if z>=-72: return np.array([0,np.interp(z,ax[:,2],ax[:,1]),z])
    return np.array([0,np.polyval(k,z),z])
def C(z,dx,dy): a=A(z);return [a[0]+dx,a[1]+dy,z]
def tr(c):return [tracks[i] for i in np.nonzero(cl==c)[0]]
def top(t):return t[np.argmax(t[:,2])]
def centre(ts,z0,z1,step=2):
    P=np.vstack(ts);out=[]
    for z in np.arange(z0,z1+.1,step):
        q=P[np.abs(P[:,2]-z)<step/2+.5]
        if len(q)>3: out.append([np.median(q[:,0]),np.median(q[:,1]),z])
    return np.array(out)
def smooth(p,s=1.5):
    p=np.array(p,float);return np.c_[gaussian_filter1d(p[:,0],s,mode='nearest'),gaussian_filter1d(p[:,1],s,mode='nearest'),gaussian_filter1d(p[:,2],s,mode='nearest')]
cst=tr(62);ml=tr(72);strr=tr(46)
m1hand=[t for t in cst if top(t)[0]<-28 and top(t)[2]>45 and -30<top(t)[1]<-8]
s1hand=[t for t in strr if top(t)[0]<-28 and top(t)[2]>40 and top(t)[1]<-22]
print(len(m1hand),len(s1hand))
cstc=np.vstack([centre(cst,-70,24),centre(m1hand,26,60),[np.median([top(t) for t in m1hand],0)]])[::-1]  # top -> down
mlc=centre(ml,-52,4)
_s1t=np.median([top(t) for t in s1hand],0)
s1c=np.vstack([centre(s1hand,10,_s1t[2]-4),[_s1t]])
vpl=np.array(R['vpl'])
print('M1 end',cstc[0],'S1 end',s1c[-1],'cst low',cstc[-1])
# ---- CST (motor, descending): left M1 -> crosses -> right lateral column -> right anterior horn -> ventral root
cst_low=cstc[-1]
CST=list(cstc)+[[cst_low[0]*.9,-36.5,-62],[cst_low[0]*.6,-39.5,-69],C(-77,-0.5,3.5),C(-80,2.5,1.0),C(-84,4.5,-1.0),C(-100,4.6,-1.0),C(-126,4.6,-1.0),C(-130,3.6,2.4),C(-132,8,5),C(-134,16,8),C(-136,24,10)]
# ---- DCML (ascending): right periphery -> dorsal root -> right dorsal column -> nucleus cuneatus -> internal arcuate (cross) -> left ML -> VPL -> S1
DC=[C(-142,26,-12),C(-140,16,-9),C(-138,6,-5.5),C(-136,2.2,-4.3),C(-110,2.4,-4.4),C(-80,2.6,-4.6),C(-74,3.2,-5.5),C(-68,4.0,-6.5),C(-63,2.5,-3.5),C(-60,0,0.5),C(-57,-2.5,2.0)]
DCML=DC+list(mlc)+[list(vpl+[0,0,1])]+list(s1c)
# ---- STT (ascending): right periphery -> dorsal horn -> anterior white commissure (cross) -> left anterolateral -> lateral brainstem -> VPL -> S1
def mlx(z):return np.interp(z,mlc[:,2],mlc[:,0]),np.interp(z,mlc[:,2],mlc[:,1])
ST=[C(-150,26,-12),C(-148,16,-9),C(-146,6,-5.5),C(-145,3.0,-3.2),C(-143,0,1.2),C(-141,-3.5,1.6),C(-138,-5.2,1.2),C(-110,-5.2,1.0),C(-80,-5.4,0.6)]
for z in range(-74,-51,4): ST.append(C(z,-7,0.5))
for z in range(-50,1,4):
    x,y=mlx(z);off=np.interp(z,[-50,-20,0],[5.5,4.5,2.5]);ST.append([x-off,y-1.0,z])
ST+= [list(vpl+[-0.8,-1.2,1.5])]+[list(p+[0,-1.6,0]) for p in s1c]
paths=dict(cst=smooth(CST,1.2).tolist(),dcml=smooth(DCML,1.2).tolist(),stt=smooth(ST,1.2).tolist())
# cord axis for schematic spinal cord
cord=[C(z,0,0) for z in range(-72,-161,-4)]
# cross sections (real brainstem outlines; schematic cord)
secs=[]
for nm,z in [('Midbrain (superior colliculus level)',-10),('Mid pons',-30),('Medulla (olive level)',-52)]:
    m=np.isin(lab[:,:,z-O[2]],bsid).astype(float)
    cs=find_contours(gaussian_filter1d(gaussian_filter1d(m,1,0),1,1),0.5)
    ring=max(cs,key=len)[::2]+O[:2]
    def at(p):
        p=np.array(p);i=np.argmin(np.abs(p[:,2]-z));return p[i][:2].tolist()
    secs.append(dict(name=nm,z=z,ring=ring.tolist(),c=A(z)[:2].tolist(),pts=dict(cst=at(paths['cst']),dcml=at(paths['dcml']),stt=at(paths['stt']))))
secs.append(dict(name='Cervical cord (C6, schematic)',z=-120,c=A(-120)[:2].tolist(),cord=True,pts=dict(cst=C(-120,4.6,-1)[:2],dcml=C(-120,2.4,-4.4)[:2],stt=C(-120,-5.2,1.0)[:2])))
def P(name,i): p=paths[name];return p[int(i*(len(p)-1))] if isinstance(i,float) else None
def near(name,z):
    p=np.array(paths[name]);return p[np.argmin(np.abs(p[:,2]-z))].tolist()
labels=[
 dict(p='cst',at=paths['cst'][0],t='Precentral gyrus (M1)',f='上運動神經元起點'),
 dict(p='cst',at=near('cst',8),t='Posterior limb, internal capsule',f='CST 下行經內囊後肢'),
 dict(p='cst',at=near('cst',-14),t='Cerebral peduncle',f='大腦腳中間 3/5'),
 dict(p='cst',at=near('cst',-34),t='Basis pontis',f='CST 散成小束'),
 dict(p='cst',at=near('cst',-60),t='Medullary pyramid',f='延腦錐體'),
 dict(p='cst',at=C(-78,0.5,3)  ,t='Pyramidal decussation',f='85–90% 在此交叉'),
 dict(p='cst',at=C(-104,4.6,-1),t='Lateral corticospinal tract',f='對側（已交叉）下行'),
 dict(p='cst',at=C(-131,4.5,3),t='Anterior horn (LMN)',f='α 運動神經元→肌肉'),
 dict(p='dcml',at=C(-122,2.4,-4.4),t='Dorsal column (fasciculus cuneatus)',f='同側上行，未交叉'),
 dict(p='dcml',at=C(-68,4,-6.5),t='Nucleus cuneatus / gracilis',f='第二級神經元'),
 dict(p='dcml',at=C(-60,0,0.5),t='Internal arcuate fibers',f='在延腦交叉'),
 dict(p='dcml',at=near('dcml',-30),t='Medial lemniscus',f='精細觸覺、本體感覺'),
 dict(p='dcml',at=vpl.tolist(),t='VPL nucleus (thalamus)',f='第三級神經元'),
 dict(p='dcml',at=paths['dcml'][-1],t='Postcentral gyrus (S1)',f='初級體感覺皮質'),
 dict(p='stt',at=C(-145,3,-3.2),t='Dorsal horn',f='第二級神經元（脊髓）'),
 dict(p='stt',at=C(-143,0,1.2),t='Anterior white commissure',f='進入後 1–2 節內交叉'),
 dict(p='stt',at=C(-100,-5.2,1),t='Anterolateral (spinothalamic) tract',f='痛、溫、粗觸覺'),
 dict(p='stt',at=near('stt',-40),t='Spinothalamic tract, brainstem',f='走在腦幹外側（示意）'),
]
json.dump(dict(paths=paths,cord=cord,secs=secs,labels=labels),open('out/paths.json','w'))
print({k:len(v) for k,v in paths.items()})
