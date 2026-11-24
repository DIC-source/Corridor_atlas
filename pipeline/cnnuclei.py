# Cranial nerve nuclei: schematic columns placed relative to the template brainstem (floor of the 4th ventricle, aqueduct, AP depth)
import numpy as np,json
O=np.array([-90,-126,-72]);lab=np.load('out/lab.npy');S=json.load(open('out/S.json'))
S=[s for s in S if s['group']!='Cranial nerve nuclei']
ids=lambda ns:[s['id'] for s in S if s['name'] in ns]
BS=np.isin(lab,ids(['Midbrain','Pons','Medulla','Cerebral peduncle (crus cerebri)']))
geo={}
for z in range(-72,1):
    zi=z-O[2];col=BS[88:93,:,zi].any(0);ys=np.nonzero(col)[0]
    if len(ys):geo[z]=(ys.max()+O[1],ys.min()+O[1],np.nonzero(BS[:,:,zi].any(1))[0])
def g(z):z=int(round(z));return geo[min(geo,key=lambda q:abs(q-z))]
def front(z):return g(z)[0]
def back(z):return g(z)[1]
def hw(z):xs=g(z)[2];return (xs.max()-xs.min())/2
# (name, colour, function group, z range, x(z), y(z), radius)
MOT,SEN,VC,AUT='#e07a3a','#4f86c6','#3fa08f','#9b6bc4'
N=[('Oculomotor nucleus (CN III)',MOT,(-9,-3),lambda z:2.5,lambda z:-30.5,1.3),
   ('Edinger–Westphal nucleus (CN III, parasympathetic)',AUT,(-6,-2),lambda z:1.0,lambda z:-31.5,0.8),
   ('Trochlear nucleus (CN IV)',MOT,(-13,-10),lambda z:2.5,lambda z:-31.5,1.0),
   ('Mesencephalic nucleus of V',SEN,(-26,-4),lambda z:4.5,lambda z:back(z)+3,0.6),
   ('Principal sensory nucleus of V',SEN,(-32,-27),lambda z:9.0,lambda z:back(z)+7,1.5),
   ('Motor nucleus of V',MOT,(-32,-27),lambda z:6.0,lambda z:back(z)+8,1.3),
   ('Abducens nucleus (CN VI)',MOT,(-41,-37),lambda z:2.5,lambda z:back(z)+1.8,1.2),
   ('Facial motor nucleus (CN VII)',MOT,(-44,-39),lambda z:7.5,lambda z:back(z)+11,1.3),
   ('Superior salivatory nucleus (CN VII)',AUT,(-45,-42),lambda z:5.0,lambda z:back(z)+6,0.7),
   ('Vestibular nuclei (CN VIII)',VC,(-50,-38),lambda z:8.0,lambda z:back(z)+2,1.8),
   ('Cochlear nuclei (CN VIII)',VC,(-48,-44),lambda z:11.5,lambda z:back(z)+4,1.1),
   ('Nucleus ambiguus (CN IX, X, XI)',MOT,(-64,-47),lambda z:6.0,lambda z:(front(z)+back(z))/2,0.8),
   ('Inferior salivatory nucleus (CN IX)',AUT,(-49,-46),lambda z:4.5,lambda z:back(z)+5,0.7),
   ('Solitary nucleus (CN VII, IX, X)',SEN,(-62,-45),lambda z:5.0,lambda z:back(z)+2.5,1.0),
   ('Dorsal motor nucleus of vagus (CN X)',AUT,(-60,-49),lambda z:3.2,lambda z:back(z)+1.6,0.9),
   ('Hypoglossal nucleus (CN XII)',MOT,(-62,-49),lambda z:1.3,lambda z:back(z)+1.6,1.0),
   ('Spinal trigeminal nucleus (CN V)',SEN,(-71,-33),lambda z:min(hw(z)-1.5,9.5),lambda z:back(z)+0.35*(front(z)-back(z)),1.1),
   ('Spinal accessory nucleus (CN XI, upper cervical)',MOT,(-72,-66),lambda z:4.5,lambda z:(front(z)+back(z))/2+1,0.8)]
X,Y,Z=np.meshgrid(*[O[i]+np.arange(lab.shape[i]) for i in range(3)],indexing='ij')
for nm,col,(z0,z1),fx,fy,r in N:
    for sd,m in [('L',-1),('R',1)]:
        i=len(S)+1;S.append(dict(id=i,name=nm,group='Cranial nerve nuclei',color=col,side=sd,note='schematic position'))
        mask=np.zeros(lab.shape,bool)
        for z in np.arange(z0,z1+0.01,0.5):
            cx,cy=m*fx(z),fy(z);zi=int(round(z-O[2]));sl=(np.abs(Z[:,:,0]-0)<1)  # placeholder
            d=(X[:,:,zi]-cx)**2+(Y[:,:,zi]-cy)**2<=max(r,0.75)**2
            mask[:,:,zi]|=d
        mask&=(BS|(lab==ids(['Fourth ventricle'])[0]) if False else BS)
        lab[mask]=i
cnt=np.bincount(lab.ravel(),minlength=len(S)+1)
for s in S:s['vox']=int(cnt[s['id']])
print([(s['name'][:28],s['side'],s['vox']) for s in S if s['group']=='Cranial nerve nuclei' and s['side']=='L'])
np.save('out/lab.npy',lab);json.dump(S,open('out/S.json','w'))
