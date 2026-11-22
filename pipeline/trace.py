import numpy as np,json
from scipy import ndimage as ndi
from skimage.graph import route_through_array
P=np.load('out/Ph.npy'); HO=np.array([-91.25,-126.25,-72.25])
Ps=ndi.gaussian_filter(P,0.7)
toI=lambda p:((np.array(p)-HO)/0.5).round().astype(int)
toM=lambda i:np.array(i)*0.5+HO
def lm(x,y,z,side=1):
    xs=sorted([x[0]*side,x[1]*side]);a=toI([xs[0],y[0],z[0]]);b=toI([xs[1],y[1],z[1]])
    sub=Ps[a[0]:b[0]+1,a[1]:b[1]+1,a[2]:b[2]+1];k=np.unravel_index(np.argmax(sub),sub.shape)
    return toM(a+np.array(k)),float(sub.max())
def route(pa,pb,pad=8):
    a,b=toI(pa),toI(pb);lo=np.maximum(np.minimum(a,b)-int(pad/0.5),0);hi=np.minimum(np.maximum(a,b)+int(pad/0.5),np.array(P.shape)-1)
    sub=Ps[lo[0]:hi[0]+1,lo[1]:hi[1]+1,lo[2]:hi[2]+1];cost=1.0/(sub/100+0.01)**2
    path,_=route_through_array(cost,tuple(a-lo),tuple(b-lo),fully_connected=True,geometric=True)
    return [list(map(float,toM(np.array(q)+lo))) for q in path]
def chain(pts):
    out=[]
    for i in range(len(pts)-1):
        r=route(pts[i],pts[i+1]);out+=r if not out else r[1:]
    return out
VES={}
for side,S in [(1,'L'),(-1,'R')]:
    L=lambda x,y,z:lm(x,y,z,side)[0]
    ica_b=L((-38,-20),(-20,10),(-68,-58)); ica_t=L((-18,-8),(-5,10),(-20,-6))
    m1e=L((-42,-30),(-5,15),(-15,-2)); m2=L((-48,-36),(-25,10),(10,25))
    acom=lm((-3,3),(5,20),(-12,0))[0]
    va=L((-20,-6),(-50,-30),(-68,-60)); vbj=lm((-4,4),(-40,-25),(-48,-38))[0]; tip=lm((-4,4),(-22,-8),(-20,-10))[0]
    p2=L((-30,-18),(-40,-25),(-15,-2)); p3=L((-25,-10),(-75,-60),(-5,10)); pcomj=L((-18,-12),(-18,-8),(-18,-10))
    sca=L((-26,-12),(-42,-26),(-36,-24))
    VES['ica_'+S.lower()]=dict(name='Internal carotid artery',side=S,pts=chain([ica_b,ica_t]))
    VES['m1_'+S.lower()]=dict(name='MCA M1 (sphenoidal)',side=S,pts=chain([ica_t,m1e]))
    VES['m2_'+S.lower()]=dict(name='MCA M2 (insular)',side=S,pts=chain([m1e,m2]))
    VES['a1_'+S.lower()]=dict(name='ACA A1',side=S,pts=chain([ica_t,acom]))
    VES['va_'+S.lower()]=dict(name='Vertebral artery (V4)',side=S,pts=chain([va,vbj]))
    VES['p1p2_'+S.lower()]=dict(name='PCA P1–P2',side=S,pts=chain([tip,p2]))
    VES['p3_'+S.lower()]=dict(name='PCA P3–P4 (calcarine/parieto-occipital)',side=S,pts=chain([p2,p3]))
    VES['pcom_'+S.lower()]=dict(name='Posterior communicating artery',side=S,pts=chain([ica_t+np.array([0,-2,-3]),pcomj]))
    VES['sca_'+S.lower()]=dict(name='Superior cerebellar artery',side=S,pts=chain([tip,sca]))
    print(S,'icab',ica_b,'icat',ica_t,'m1e',m1e,'m2',m2,'va',va,'p2',p2,'p3',p3,'pcomj',pcomj,'sca',sca)
acom=lm((-3,3),(5,20),(-12,0))[0];vbj=lm((-4,4),(-40,-25),(-48,-38))[0];tip=lm((-4,4),(-22,-8),(-20,-10))[0]
g1=lm((-6,6),(25,42),(-2,15))[0];g2=lm((-6,6),(10,30),(25,42))[0];g3=lm((-6,6),(-30,-10),(30,48))[0];g4=lm((-6,6),(-55,-40),(15,35))[0]
VES['a2']=dict(name='ACA A2 – pericallosal artery',side='',pts=chain([acom,g1,g2,g3,g4]))
VES['ba']=dict(name='Basilar artery',side='',pts=chain([vbj,tip]))
print('acom',acom,'vbj',vbj,'tip',tip,'genu',g1,g2,g3,g4)
for k,v in VES.items():
    p=np.array(v['pts']);v['minP']=float(min(Ps[tuple(toI(q))] for q in p));v['len']=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    print(k,v['name'],len(p),'len %.0f'%v['len'],'minP %.1f'%v['minP'])
json.dump(VES,open('out/arteries.json','w'))
