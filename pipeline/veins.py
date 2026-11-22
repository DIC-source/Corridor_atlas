# Dural sinuses and deep veins: centerlines placed from template geometry (brain surface, CC, 3rd ventricle, midbrain).
import numpy as np,json
from scipy.interpolate import splprep,splev
from tt import load_tt
O=np.array([-90,-126,-72]);br=np.load('out/brainmask.npy');lab=np.load('out/lab.npy');S=json.load(open('out/S.json'))
sid=lambda n:[s['id'] for s in S if s['name']==n]
def inside(p):
    v=np.round(np.array(p)-O).astype(int)
    return np.all(v>=0) and np.all(v<br.shape) and br[tuple(v)]
def ray(c,d,off):
    d=np.array(d,float);d/=np.linalg.norm(d);p=np.array(c,float);last=None;t=0
    while t<150:
        if inside(p+d*t): last=t
        t+=0.5
    return p+d*((last if last is not None else 0)+off)
def smooth(pts,step=1.0,s=None):
    p=np.array(pts,float);k=min(3,len(p)-1);tck,u=splprep(p.T,s=len(p)*1.5 if s is None else s,k=k)
    L=np.linalg.norm(np.diff(p,axis=0),axis=1).sum();uu=np.linspace(0,1,max(4,int(L/step)));return np.array(splev(uu,tck)).T
OUT={}
# torcula: posterior brain surface at x=0, z=0
torc=None
# superior sagittal sinus: rays over the top in the mid-sagittal plane (mask |x|<=6)
def inside_mid(p):
    return any(inside([x,p[1],p[2]]) for x in (-6,-3,0,3,6))
def ray_mid(c,d,off):
    # march outward; only stop at the LAST exit from the brain (skips CSF gaps near the start)
    d=np.array(d,float);d/=np.linalg.norm(d);p=np.array(c,float);last=None;t=0
    while t<150:
        if inside_mid(p+d*t): last=t
        t+=0.5
    return p+d*((last if last is not None else 0)+off)
torc=None
c=np.array([0,-20,5.0]);sss=[]
for th in np.radians(np.arange(8,196,5)):   # angle from +y (anterior) toward +z then posterior
    sss.append(ray_mid(c,[0,np.cos(th),np.sin(th)],4.0))
torc=ray_mid([0,-70,-4],[0,-1,0.12],4.0)
sss.append(torc);OUT['sss']=dict(name='Superior sagittal sinus',side='',r=(2.5,4.5),pts=smooth(sss))
# splenium & Galen from HCP corpus callosum streamlines
tracks,cl,names=load_tt()
cc=np.vstack([tracks[i] for i in np.nonzero(np.isin(cl,[85,86,88]))[0]]);cc=cc[(np.abs(cc[:,0])<1.5)&(cc[:,2]>-2)&(cc[:,2]<40)&(cc[:,1]>-55)&(cc[:,1]<40)]
spl=cc[np.argmin(cc[:,1])];galen=np.array([0,spl[1]-5,spl[2]-5])
OUT['straight']=dict(name='Straight sinus',side='',r=(2.5,2.8),pts=smooth([galen,(galen*0.6+torc*0.4)+[0,0,-2],torc],s=0))
OUT['galen']=dict(name='Great cerebral vein (Galen)',side='',r=(3,3),pts=smooth([galen+[0,4,3],galen],s=0))
# inferior sagittal sinus: ~7 mm above the top of the corpus callosum
iss=[]
for y in range(20,int(spl[1])+2,-3):
    q=cc[np.abs(cc[:,1]-y)<1.5]
    if len(q): iss.append([0,y,q[:,2].max()+7])
iss.append(galen+[0,3,4]);OUT['iss']=dict(name='Inferior sagittal sinus',side='',r=(1.3,1.6),pts=smooth(iss))
# internal cerebral veins: along the roof of the third ventricle, Monro -> Galen
v3=np.array(np.nonzero(np.isin(lab,sid('Third ventricle')))).T+O
for sd,m in [('l',-1),('r',1)]:
    pts=[[2.5*m,2,10]]
    for y in range(-2,int(galen[1])+4,-4):
        q=v3[np.abs(v3[:,1]-y)<1.5]
        if len(q): pts.append([2.5*m,y,q[:,2].max()+3])
    pts.append(galen+[1.2*m,3,1]);OUT['icv_'+sd]=dict(name='Internal cerebral vein',side=sd.upper(),r=(1.2,1.4),pts=smooth(pts))
# basal vein of Rosenthal: anterior perforated substance -> around the cerebral peduncle -> Galen
mb=np.array(np.nonzero(np.isin(lab,sid('Midbrain')))).T+O
for sd,m in [('l',-1),('r',1)]:
    def lat(y,z):
        q=mb[(np.abs(mb[:,1]-y)<1.5)&(np.abs(mb[:,2]-z)<1.5)&(mb[:,0]*m>0)];return np.abs(q[:,0]).max() if len(q) else 12
    pts=[[20*m,4,-14],[(lat(-14,-12)+4)*m,-14,-12],[(lat(-24,-8)+4)*m,-24,-8],[(lat(-34,-2)+3)*m,-34,-2],galen+[3*m,2,-1]]
    OUT['bvr_'+sd]=dict(name='Basal vein of Rosenthal',side=sd.upper(),r=(1.2,1.5),pts=smooth(pts,s=0))
# transverse and sigmoid sinuses: rays around the posterolateral brain surface at the tentorial level, then down to the jugular foramen
for sd,m in [('l',-1),('r',1)]:
    ts=[torc]
    for ph in np.radians(np.arange(10,71,6)):
        z=torc[2]-0.3*np.degrees(ph);ts.append(ray([m*6,-62,z],[m*np.sin(ph),-np.cos(ph),0],4.0))
    J=ts[-1];sg=[J]
    for f in np.linspace(0.2,0.75,4):
        z=J[2]+(-50-J[2])*f;y=J[1]+(-38-J[1])*f;sg.append(ray([m*22,y,z],[m,-0.25,0],3.5))
    sg.append(np.array([m*40,-36,-54.0]))
    OUT['ts_'+sd]=dict(name='Transverse sinus',side=sd.upper(),r=(3.5,4),pts=smooth(ts))
    OUT['sig_'+sd]=dict(name='Sigmoid sinus',side=sd.upper(),r=(3.8,4),pts=smooth(sg))
for k,v in OUT.items():
    v['pts']=[list(map(float,p)) for p in v['pts']];p=np.array(v['pts'])
    print(k,v['name'],len(p),'start',p[0].round(0),'end',p[-1].round(0))
json.dump(OUT,open('out/veins.json','w'))
