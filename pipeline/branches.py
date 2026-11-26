# Named distal arterial branches (population course): MCA M2 trunks + M3/M4 cortical branches,
# ACA cortical branches, PCA cortical/choroidal branches, AComm, ophthalmic, anterior spinal.
# Course = sulcal/fissural landmarks from the cortical parcellation, refined onto the Mouches
# population vessel-probability ridge where one exists.  Output: out/branches.json
import numpy as np,json,sys
from scipy import ndimage as ndi
from skimage.graph import route_through_array
from sulci import adj,line,surf,DEPTH,NL,lid,O,SH
A=json.load(open('out/arteries.json'))
P=np.load('out/Ph.npy');HO=np.array([-91.25,-126.25,-72.25]);Ps=ndi.gaussian_filter(P,1.0)
toI=lambda p:np.clip(((np.array(p)-HO)/0.5).round().astype(int),0,np.array(P.shape)-1)
toM=lambda i:np.array(i)*0.5+HO
def pval(p):return float(Ps[tuple(toI(p))])
def route(pa,pb,pad=3.0):
    a,b=toI(pa),toI(pb);lo=np.maximum(np.minimum(a,b)-int(pad/0.5),0);hi=np.minimum(np.maximum(a,b)+int(pad/0.5),np.array(P.shape)-1)
    sub=Ps[lo[0]:hi[0]+1,lo[1]:hi[1]+1,lo[2]:hi[2]+1];cost=1.0/(sub/100+0.03)
    path,_=route_through_array(cost,tuple(a-lo),tuple(b-lo),fully_connected=True,geometric=True)
    return [toM(np.array(q)+lo) for q in path]
def refine(wp,pad=2.5):
    """route between consecutive waypoints along the probability ridge (bounded corridor), then resample"""
    out=[np.array(wp[0],float)]
    for a,b in zip(wp[:-1],wp[1:]):out+=route(a,b,pad)[1:]
    return sm(np.array(out))
from scipy.interpolate import splprep,splev
def sm(p,step=1.0,k=1.2):
    _,u=np.unique(np.round(p,2),axis=0,return_index=True);p=p[np.sort(u)]
    if len(p)<5:return p
    L=np.linalg.norm(np.diff(p,axis=0),axis=1).sum();tck,_=splprep(p.T,s=len(p)*k,k=3)
    return np.array(splev(np.linspace(0,1,max(6,int(L/step))),tck)).T
def lab_pt(i,side,near=None,depth=1.5):
    w=np.argwhere(NL==lid(i,side))+O;w=w[DEPTH[tuple((w-O).T)]<=3]
    p=w.mean(0) if near is None else w[np.argmin(np.linalg.norm(w-np.array(near),axis=1))]
    return surf(p,depth)
def at(L,axis,v):
    """interpolate a landmark line at coordinate v along axis"""
    i=np.argsort(L[:,axis]);return np.array([np.interp(v,L[i,axis],L[i,k]) for k in range(3)])
OPF=[16,18,22,20,29,17]
OUT={}
def add(key,name,side,wp,r,parent,color,note,pad=2.5,ref=True):
    pts=refine(wp,pad) if ref else np.array(wp,float)
    OUT[key+'_'+side if side else key]=dict(name=name,side=side.upper(),pts=np.round(pts,2).tolist(),r=r,parent=parent,color=color,note=note,wp=np.round(np.array(wp,float),1).tolist())
MC,AC,PC,OT='#ee6a50','#f08b3c','#b5487f','#d04848'
for s,m in [('l',-1),('r',1)]:
    lat=lambda w:w[m*w[:,0]>22]           # lateral convexity only
    Syl=line(lat(adj([28],OPF,s,3)),1,25,-42,4)
    SCS=line(adj([31],OPF,s,30,3),1,25,-28,4)
    ICS=line(adj([31],[28,30],s,30,3),1,15,-28,4)
    CS =line(lat(adj([22],[20],s,3)),2,10,68,5)
    PreCS=line(lat(adj([22],[2,16],s,3)),2,12,62,5)
    PoCS=line(lat(adj([20],[29],s,3)),2,20,58,5)
    IPS=line(lat(adj([27],[29,6],s,3)),1,-42,-80,5)
    STS=line(lat(adj([28],[13],s,3)),1,8,-58,4)
    IFS=line(lat(adj([16,18],[25],s,3)),1,10,42,5)
    ASR=line(lat(adj([18],[16],s,3)),2,0,25,4)          # anterior border of pars opercularis (ascending ramus)
    SMp=line(lat(adj([29],[6],s,3)),2,20,52,5)
    S_=lambda p,d=1.5:surf(p,d)
    m1=np.array(A['m1_'+s]['pts']);lim=m1[-1]
    sc=lambda y:at(SCS,1,y);ic=lambda y:at(ICS,1,y);sy=lambda y:S_(at(Syl,1,y))
    # ---- M2 trunks (insular)
    gen=lim+np.array([m*2,-2,4])
    add('m2s','MCA M2 – superior trunk (insular)',s,[lim,gen,sc(8),sc(-6),sc(-16)],[1.3,1.0],'a_m1_'+s,MC,'population course on the insula (short gyri to the superior circular sulcus)')
    add('m2','MCA M2 – inferior trunk (insular)',s,[lim,ic(2),ic(-12),ic(-22),(sc(-24)+ic(-26))/2],[1.3,1.0],'a_m1_'+s,MC,'population course on the insula (long gyri toward the posterior insular point)')
    post=(sc(-24)+ic(-26))/2
    def opc(a,b):return (a*0.45+b*0.55)    # mid-operculum (inner opercular surface)
    # ---- superior-trunk cortical branches (M3 over the fronto-parietal operculum, M4 in the sulcus)
    lat_of=lambda L,v:S_(at(L,2,v))
    of_end=lab_pt(10,s,near=[m*40,38,-14]);po=lab_pt(17,s)
    add('mca_of','MCA orbitofrontal (lateral fronto-basal) artery',s,[gen,sc(14),opc(sc(18),sy(22)),sy(22),po,of_end],[0.9,0.6],'a_m2s_'+s,MC,'M3 over the pars orbitalis to the lateral orbital gyri')
    ifs=[S_(p) for p in IFS[IFS[:,1].argsort()]]
    add('mca_pf','MCA prefrontal (candelabra) artery',s,[sc(10),opc(sc(12),sy(14)),sy(14),S_(at(ASR,2,14)),ifs[0],ifs[len(ifs)//2],ifs[-1]],[0.9,0.6],'a_m2s_'+s,MC,'M3 out of the anterior Sylvian fissure, up the ascending ramus and forward along the inferior frontal sulcus')
    pc=[lat_of(PreCS,z) for z in (18,32,46,58)]
    add('mca_prc','MCA precentral (pre-Rolandic) artery',s,[sc(2),opc(sc(2),sy(4)),sy(4)]+pc,[0.9,0.6],'a_m2s_'+s,MC,'M4 in the precentral sulcus')
    cc=[lat_of(CS,z) for z in (16,30,44,58,66)]
    add('mca_cen','MCA central (Rolandic) artery',s,[sc(-8),opc(sc(-8),sy(-6)),sy(-6)]+cc,[0.9,0.6],'a_m2s_'+s,MC,'M4 in the central sulcus (hand/face motor-sensory cortex)')
    ip=[S_(p) for p in IPS[IPS[:,1].argsort()[::-1]]]
    pcs=[lat_of(PoCS,z) for z in (26,38,50)]
    add('mca_ap','MCA anterior parietal artery',s,[sc(-16),opc(sc(-16),sy(-18)),sy(-18)]+pcs+ip[:2],[0.9,0.6],'a_m2s_'+s,MC,'M4 in the postcentral sulcus, then the anterior intraparietal sulcus')
    # ---- inferior-trunk cortical branches
    smp=[lat_of(SMp,z) for z in (26,38,50)]
    add('mca_pp','MCA posterior parietal artery',s,[post,opc(post,sy(-34)),sy(-34)]+smp,[0.9,0.6],'a_m2_'+s,MC,'M4 over the supramarginal gyrus toward the intraparietal sulcus')
    ang=lab_pt(6,s,near=[m*46,-68,32])
    sts=lambda y:S_(at(STS,1,y))
    add('mca_ang','MCA angular artery',s,[post,opc(post,sy(-40)),sy(-40),sts(-52),S_((sts(-52)+ang)/2),ang],[0.9,0.6],'a_m2_'+s,MC,'M4 from the end of the Sylvian fissure along the posterior STS into the angular gyrus')
    lo=lab_pt(9,s,near=[m*50,-76,2])
    add('mca_to','MCA temporo-occipital artery',s,[post,opc(post,sy(-36)),sy(-36),sts(-46),sts(-56),lo],[0.85,0.6],'a_m2_'+s,MC,'M4 along the posterior STS to the lateral occipital cortex')
    stg=lambda y:S_((sy(y)+sts(y))/2)
    mtg=lambda y,dz=-10:S_(sts(y)+np.array([m*2,0,dz]))
    add('mca_pt','MCA posterior temporal artery',s,[ic(-20),opc(ic(-20),sy(-24)),sy(-24),stg(-30),sts(-36),mtg(-44)],[0.85,0.6],'a_m2_'+s,MC,'M3 over the temporal operculum, M4 across the posterior STG/STS to the middle temporal gyrus')
    add('mca_mt','MCA middle temporal artery',s,[ic(-6),opc(ic(-6),sy(-8)),sy(-8),stg(-12),sts(-16),mtg(-20)],[0.85,0.6],'a_m2_'+s,MC,'M4 across the mid-STG to the middle temporal gyrus')
    add('mca_at','MCA anterior temporal artery',s,[ic(6),opc(ic(6),sy(6)),stg(2),sts(-2),mtg(-4,-12)],[0.85,0.6],'a_m1_'+s,MC,'often from M1: over the anterior STG to the anterior middle temporal gyrus')
    tp=lab_pt(28,s,near=[m*42,18,-30])
    q=m1[int(len(m1)*0.55)]
    add('mca_tp','MCA temporopolar artery',s,[q,q+np.array([m*4,4,-6]),tp],[0.8,0.6],'a_m1_'+s,MC,'from M1 to the temporal pole',pad=2.5)

# ---------------- medial surface: ACA and PCA cortical branches ----------------
def med_pt(i,side,near,xmax=12):
    w=np.argwhere(NL==lid(i,side))+O;w=w[np.abs(w[:,0])<=xmax]
    return w[np.argmin(np.linalg.norm(w-np.array(near),axis=1))].astype(float)
def medline(A_,B_,side,axis,lo,hi,step=5,xmax=14):
    w=adj(A_,B_,side,60);w=w[np.abs(w[:,0])<=xmax];return line(w,axis,lo,hi,step)
a2=np.array(A['a2']['pts'])
def a2at(y,upper=True):
    c=a2[np.abs(a2[:,1]-y)<3]
    return (c[np.argmax(c[:,2])] if upper else c[np.argmin(c[:,2])]) if len(c) else a2[np.argmin(np.abs(a2[:,1]-y))]
for s,m in [('l',-1),('r',1)]:
    xo=lambda p,dx=3:np.array([m*dx,p[1],p[2]],float)
    st=a2at(22,False)
    add('aca_mof','ACA medial orbitofrontal artery',s,[st,xo(st+[0,6,-8],3),med_pt(12,s,[m*5,36,-20]),med_pt(12,s,[m*6,50,-22])],[0.7,0.5],'a_a2_'+s+'@6',AC,'first cortical branch of A2: gyrus rectus and medial orbital surface')
    g=a2at(32,False)
    add('aca_fp','ACA frontopolar artery',s,[g,xo(g+[0,8,4],3),med_pt(26,s,[m*5,58,8]),med_pt(26,s,[m*6,66,20])],[0.75,0.5],'a_a2_'+s,AC,'from A2 below the genu, forward over the medial frontal surface to the frontal pole')
    pq=a2at(-12)
    add('aca_pcl','ACA paracentral artery',s,[pq,xo(pq+[0,-4,14],4),med_pt(15,s,[m*5,-22,58]),med_pt(15,s,[m*5,-30,70])],[0.75,0.5],'a_a2_'+s,AC,'pericallosal/callosomarginal branch to the paracentral lobule (leg motor-sensory cortex)')
    pr=a2at(-34)
    add('aca_pcu','ACA precuneal (internal parietal) artery',s,[pr,xo(pr+[0,-8,10],4),med_pt(23,s,[m*5,-55,40]),med_pt(23,s,[m*6,-62,56])],[0.7,0.5],'a_a2_'+s,AC,'distal pericallosal branch to the precuneus')
    p12=np.array(A['p1p2_'+s]['pts']);p3=np.array(A['p3_'+s]['pts']);pe=p3[-1]
    calc=medline([19],[3,11,19],s,1,-62,-100,6)
    add('pca_calc','PCA calcarine artery',s,[pe]+[c for c in calc if c[1]<pe[1]-4],[0.8,0.55],'a_p3_'+s,PC,'P4 in the calcarine sulcus to the occipital pole (primary visual cortex)')
    pos=medline([3],[23],s,2,8,48,6)
    add('pca_po','PCA parieto-occipital artery',s,[pe]+[c for c in pos if c[2]>pe[2]+3],[0.8,0.55],'a_p3_'+s,PC,'P4 up the parieto-occipital sulcus (cuneus/precuneus)')
    q=p12[int(len(p12)*0.75)]
    add('pca_at','PCA anterior temporal artery',s,[q,q+[m*5,2,-8],med_pt(14,s,[m*26,-18,-26],40),med_pt(5,s,[m*32,-14,-36],45)],[0.75,0.5],'a_p1p2_'+s,PC,'from P2: parahippocampal and anterior inferior temporal surface',pad=2.0)
    q=p3[int(len(p3)*0.2)]
    add('pca_pt','PCA posterior temporal artery',s,[q,q+[m*8,-8,-6],med_pt(5,s,[m*32,-46,-18],45),med_pt(5,s,[m*34,-66,-14],45)],[0.75,0.5],'a_p3_'+s,PC,'from P2/P3: fusiform and inferior temporo-occipital surface',pad=2.0)
    q=p12[int(len(p12)*0.35)]
    add('pca_mpch','Medial posterior choroidal artery',s,[q,[m*13,-26,-14],[m*9,-34,-6],[m*4,-32,4],[m*2.5,-20,10],[m*2.5,-6,12]],[0.55,0.45],'a_p1p2_'+s,PC,'around the midbrain to the quadrigeminal cistern, then forward in the velum interpositum (roof of the third ventricle) to the foramen of Monro',ref=False)
    q=p12[int(len(p12)*0.9)]
    add('pca_lpch','Lateral posterior choroidal artery',s,[q,[m*22,-30,-8],[m*26,-34,0],[m*28,-38,6]],[0.55,0.45],'a_p1p2_'+s,PC,'from P2 laterally through the choroidal fissure to the choroid plexus of the atrium / temporal horn',ref=False)
    q=p12[int(len(p12)*0.15)]
    add('pca_tp','Thalamoperforating arteries (P1)',s,[q,[q[0]+m*1,q[1]-2,q[2]+6],[m*5,-16,-4],[m*7,-17,4]],[0.45,0.4],'a_p1p2_'+s,PC,'posterior perforated substance to the thalamus/midbrain (represented by one vessel; normally several)',ref=False)
    # ophthalmic artery: from the supraclinoid ICA below the optic nerve, through the optic canal into the orbit
    ic=np.array(A['ica_'+s]['pts']);c=ic[(ic[:,2]>-27)&(ic[:,2]<-21)&(ic[:,1]>-6)];q=c[np.argmax(c[:,1])] if len(c) else ic[-8]
    add('oph','Ophthalmic artery',s,[q,[m*14,8,-25],[m*16,16,-26],[m*22,28,-28],[m*28,42,-30]],[0.7,0.6],'a_ica_'+s,OT,'from the supraclinoid ICA (below the optic nerve), through the optic canal into the orbit',ref=False)
# anterior communicating artery and anterior spinal artery (midline)
c0=np.array(A['a1_l']['pts'][-1])
OUT['acom']=dict(name='Anterior communicating artery (AComm)',side='',pts=[[c0[0]-2.6,c0[1],c0[2]],[c0[0],c0[1]+0.4,c0[2]],[c0[0]+2.6,c0[1],c0[2]]],r=[0.8,0.8],parent=None,color=AC,note='joins both A1s at the A1–A2 junction (aneurysm site)',wp=[])
vl=np.array(A['va_l']['pts']);vr=np.array(A['va_r']['pts'])
ql=vl[np.argmin(np.abs(vl[:,2]+54))];qr=vr[np.argmin(np.abs(vr[:,2]+54))]
add('asa_l','Anterior spinal artery (left root)','',[ql,[ql[0]*0.4,ql[1]-2,-58],[0,ql[1]-3,-62]],[0.45,0.45],'a_va_l',OT,'from the V4 segment to the midline anterior spinal artery',ref=False)
add('asa_r','Anterior spinal artery (right root)','',[qr,[qr[0]*0.4,qr[1]-2,-58],[0,qr[1]-3,-62]],[0.45,0.45],'a_va_r',OT,'from the V4 segment to the midline anterior spinal artery',ref=False)
y0=(ql[1]+qr[1])/2-3
add('asa','Anterior spinal artery','',[[0,y0,-62],[0,y0-2,-70],[0,y0-4,-80],[0,y0-5,-90]],[0.5,0.45],None,OT,'midline, in the anterior median fissure of the medulla and cord',ref=False)
json.dump(OUT,open('out/branches.json','w'))
for k,v in OUT.items():
    p=np.array(v['pts']);L=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
    print(k,len(p),'len %.0f'%L,'meanP %.1f'%np.mean([pval(q) for q in p]))
