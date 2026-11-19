import numpy as np,json
from tt import load_tt
tracks,cl,names=load_tt()
lab=np.load('out/lab.npy');S=json.load(open('out/S.json'))
O=np.array([-90,-126,-72])
ids={s['name']+'|'+s['side']:s['id'] for s in S}
def cen(m):return np.array(np.nonzero(m)).mean(1)+O
bs=np.isin(lab,[ids['Midbrain|'],ids['Pons|'],ids['Medulla|']])
axis=[]
for z in range(-72,-5,2):
    m=bs[:,:,z-O[2]]
    if m.sum()>5: axis.append([*(np.array(np.nonzero(m)).mean(1)+O[:2]),z])
axis=np.array(axis);print('bs axis low',axis[:3],'high',axis[-2:])
vpl=cen(lab==ids['Ventral posterolateral (VPL)|L']);print('VPL L',vpl)
def tr(c):return [tracks[i] for i in np.nonzero(cl==c)[0]]
def centre(ts,z0,z1,step=2,sel=None):
    P=np.vstack(ts);out=[]
    for z in np.arange(z0,z1+.1,step):
        q=P[np.abs(P[:,2]-z)<step/2+.5]
        if len(q)>3: out.append([np.median(q[:,0]),np.median(q[:,1]),z])
    return np.array(out)
cst=tr(62);ml=tr(72);strr=tr(46)
# hand-area streamlines: top endpoint lateral
def top(t):return t[np.argmax(t[:,2])]
hand=lambda ts:[t for t in ts if top(t)[0]<-28 and top(t)[2]>40]
print('cst',len(cst),'hand',len(hand(cst)),'ml',len(ml),'str',len(strr),'strhand',len(hand(strr)))
c1=centre(cst,-70,20); c2=centre(hand(cst),22,70)
tops=np.array([top(t) for t in hand(cst)]);print('CST top median',np.median(tops,0))
m1=centre(ml,-70,4);print('ML range',m1[[0,-1]])
s2=centre(hand(strr),8,70);tops2=np.array([top(t) for t in hand(strr)]);print('STR top median',np.median(tops2,0),s2[[0,-1]])
json.dump(dict(axis=axis.tolist(),vpl=vpl.tolist(),cst=np.vstack([c1,c2,[np.median(tops,0)]]).tolist(),ml=m1.tolist(),str=np.vstack([s2,[np.median(tops2,0)]]).tolist()),open('out/paths_raw.json','w'))
