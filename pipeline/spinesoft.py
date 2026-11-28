# Soft tissue around the spine for the textbook-style plate and the synthetic MRI, from BodyParts3D (DBCLS, CC BY-SA 2.1 JP),
# the same generic reference body as the vertebrae. Each part is flipped and moved with exactly the transform spinebp.py uses
# for the bones (C1 canal centre to the atlas cord), cropped to the spine's height and decimated.  -> docs/spine-soft.wasm
import numpy as np,struct,json,gzip,sys,re
import fast_simplification as fs
from scipy import ndimage as ndi
SRC=sys.argv[1] if len(sys.argv)>1 else '../data/raw/bodyparts3d/stl/'
def load(p):
    b=open(p,'rb').read()
    if b[:5]==b'solid' and b'facet' in b[:400]:
        t=np.array(re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)',b),float).reshape(-1,3,3)
    else:
        n=struct.unpack('<I',b[80:84])[0];a=np.frombuffer(b[84:84+n*50],dtype=np.dtype([('n','<3f4'),('v','<9f4'),('a','<u2')]));t=a['v'].reshape(-1,3,3).astype(float)
    v,inv=np.unique(np.round(t.reshape(-1,3),4),axis=0,return_inverse=True);f=inv.reshape(-1,3)
    f=f[(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])];return v,f
flip=lambda v:v*np.array([-1,-1,1])
def samples(v,f,k=4):
    a,b,c=v[f[:,0]],v[f[:,1]],v[f[:,2]];r=np.random.default_rng(0).random((len(f),k,2));m=r.sum(2)>1;r[m]=1-r[m]
    return np.concatenate([v,(a[:,None]+(b-a)[:,None]*r[...,:1]+(c-a)[:,None]*r[...,1:]).reshape(-1,3)])
def canal_c1():
    v,f=load(SRC+'FMA12519.stl');v=flip(v);P=samples(v,f);z=np.median(P[:,2]);res=0.5
    q=P[(P[:,2]>=z-5)&(P[:,2]<=z+5)][:,:2];o=q.min(0)-6;sh=np.ceil((q.max(0)+6-o)/res).astype(int)+1;g=np.zeros(sh,bool)
    ij=np.round((q-o)/res).astype(int);g[ij[:,0],ij[:,1]]=True;g=ndi.binary_closing(ndi.binary_dilation(g,iterations=2),iterations=3)
    hl,n=ndi.label(ndi.binary_fill_holes(g)&~g);best=max(range(1,n+1),key=lambda i:(hl==i).sum());c=np.argwhere(hl==best).mean(0)*res+o
    return np.array([c[0],c[1],z])
c1=canal_c1()
place=lambda v:(v-c1)+np.array([0,-46.8,-82.0])   # identical to spinebp.py (no rotation)
# (FMA id, name, class, Chinese note). class: mus | art | vein | visc | air | gut | skin | bone2 | fat
P=[]
def pair(r,l,name,cl,zh):P.extend([(r,'Right '+name,cl,zh),(l,'Left '+name,cl,zh)])
pair(22342,22343,'psoas major','mus','腰大肌：腰神經叢穿行其中；側方入路（XLIF）要穿過');pair(22348,22349,'quadratus lumborum','mus','腰方肌')
pair(22878,22879,'multifidus','mus','多裂肌：最內側的背肌，貼著椎板與棘突；後方入路先剝離它')
pair(22751,22753,'longissimus thoracis','mus','最長肌（豎脊肌中間柱）：Wiltse 入路走多裂肌與最長肌之間')
pair(22757,22758,'longissimus cervicis','mus','頸最長肌');pair(22754,22756,'longissimus capitis','mus','頭最長肌')
pair(22740,22741,'iliocostalis lumborum','mus','腰髂肋肌（豎脊肌外側柱）');pair(22742,22743,'iliocostalis thoracis','mus','胸髂肋肌');pair(22744,22745,'iliocostalis cervicis','mus','頸髂肋肌')
pair(22779,22780,'spinalis thoracis','mus','胸棘肌（豎脊肌內側柱）');pair(22781,22782,'spinalis cervicis','mus','頸棘肌')
pair(22872,22873,'semispinalis thoracis','mus','胸半棘肌');pair(22874,22875,'semispinalis cervicis','mus','頸半棘肌：C2 棘突上的主要伸肌附著，保留可減少頸椎後凸')
pair(22876,22877,'semispinalis capitis','mus','頭半棘肌');pair(22728,22729,'splenius capitis','mus','頭夾肌');pair(22726,22727,'splenius cervicis','mus','頸夾肌')
pair(33586,33587,'trapezius (descending part)','mus','斜方肌上部');pair(33584,33585,'trapezius (transverse part)','mus','斜方肌中部');pair(33581,33583,'trapezius (ascending part)','mus','斜方肌下部')
pair(13381,13382,'rhomboid major','mus','大菱形肌');pair(13383,13384,'rhomboid minor','mus','小菱形肌');pair(32540,32541,'levator scapulae','mus','提肩胛肌')
pair(13358,13359,'latissimus dorsi','mus','闊背肌');pair(13403,13404,'serratus posterior superior','mus','上後鋸肌');pair(13405,13406,'serratus posterior inferior','mus','下後鋸肌')
pair(13392,13393,'scalenus anterior','mus','前斜角肌：膈神經在其前面下行；臂神經叢在前、中斜角肌之間');pair(13390,13391,'scalenus medius','mus','中斜角肌');pair(13388,13389,'scalenus posterior','mus','後斜角肌')
pair(13408,13409,'sternocleidomastoid','mus','胸鎖乳突肌：前方頸椎入路（Smith–Robinson）在它內側進入')
for i,(r,l) in enumerate([(46283,46284),(46285,46286),(46287,46288)]):pair(r,l,'longus colli','mus','頸長肌：ACDF 時向兩側剝離，交感鏈在它表面（外側剝太多 → Horner）')
pair(46309,46310,'longus capitis','mus','頭長肌')
pair(13336,13337,'external oblique','mus','腹外斜肌');pair(13892,13893,'internal oblique','mus','腹內斜肌');pair(22344,22345,'transversus abdominis','mus','腹橫肌');pair(13377,13378,'rectus abdominis','mus','腹直肌')
pair(22322,22323,'iliacus','mus','髂肌');pair(22328,22329,'gluteus maximus','mus','臀大肌');pair(22330,22331,'gluteus medius','mus','臀中肌');pair(22340,22341,'piriformis','mus','梨狀肌：坐骨神經在其下緣出骨盆')
pair(13398,13399,'serratus anterior','mus','前鋸肌');pair(32547,32548,'infraspinatus','mus','棘下肌');pair(13414,13415,'subscapularis','mus','肩胛下肌')
pair(32534,32535,'obliquus capitis superior','mus','頭上斜肌（枕下三角）');pair(32536,32537,'obliquus capitis inferior','mus','頭下斜肌：C2 棘突到 C1 橫突，枕下三角下界')
P.append((13295,'Diaphragm','mus','橫膈：左右膈腳附著在 L1–L3 椎體前面'))
pair(23089,23090,'lumbar rotatores','mus','腰迴旋肌');P.append((23083,'Thoracic rotatores','mus','胸迴旋肌'))
P.append((71444,'Lateral lumbar intertransversarii','mus','腰橫突間外側肌'));P.append((76775,'Medial lumbar intertransversarii','mus','腰橫突間內側肌'))
P.append((3784,'Descending aorta','art','降主動脈：在椎體左前方；節段動脈從它後面發出'));P.append((3768,'Arch of aorta','art','主動脈弓'))
pair(14765,14766,'common iliac artery','art','總髂動脈：L4–L5 前方；ALIF 時的主要風險');pair(18806,18807,'external iliac artery','art','外髂動脈');pair(18809,18810,'internal iliac artery','art','內髂動脈')
pair(3941,4058,'common carotid artery','art','總頸動脈：頸動脈鞘內，ACDF 時向外側牽開');pair(3953,4694,'subclavian artery','art','鎖骨下動脈：椎動脈由此發出')
pair(14752,14753,'renal artery','art','腎動脈');P.append((50737,'Celiac trunk','art','腹腔幹（T12–L1 前方）'));P.append((14749,'Superior mesenteric artery','art','腸繫膜上動脈'))
P.append((10951,'Inferior vena cava','vein','下腔靜脈：在椎體右前方'))
pair(21387,21388,'common iliac vein','vein','總髂靜脈：左總髂靜脈橫過 L5 前方，ALIF/OLIF 最容易撕裂');pair(18885,18886,'external iliac vein','vein','外髂靜脈');pair(18887,18888,'internal iliac vein','vein','內髂靜脈')
pair(4754,4762,'internal jugular vein','vein','內頸靜脈：頸動脈鞘外側');pair(4751,4761,'brachiocephalic vein','vein','頭臂靜脈');pair(14335,14336,'renal vein','vein','腎靜脈：左腎靜脈橫過主動脈前方')
P.append((7131,'Esophagus','gut','食道：頸椎前方入路時要向內側牽開，避免穿孔'));P.append((7394,'Trachea','air','氣管'))
pair(7204,7205,'kidney','visc','腎臟');pair(15629,15630,'adrenal gland','visc','腎上腺');P.append((7197,'Liver','visc','肝臟'));P.append((7196,'Spleen','visc','脾臟'))
P.append((7206,'Duodenum','gut','十二指腸：第三段橫過 L3 前方'));P.append((7148,'Stomach','gut','胃'));P.append((14544,'Rectum','gut','直腸'));P.append((15900,'Urinary bladder','visc','膀胱'))
P.append((55099,'Thyroid cartilage','bone2','甲狀軟骨（約 C4–C5 高度）'));P.append((52749,'Hyoid bone','bone2','舌骨（約 C3 高度）'))
pair(16586,16587,'hip bone','bone2','髖骨：髂嵴約在 L4–L5 高度');pair(13395,13396,'scapula','bone2','肩胛骨');pair(13322,13323,'clavicle','bone2','鎖骨');P.append((52735,'Occipital bone','bone2','枕骨'))
P.append((7163,'Skin','skin','皮膚'))
TGT={'mus':1400,'art':900,'vein':900,'visc':1600,'gut':1200,'air':900,'bone2':1600,'skin':42000}
ZT,ZB=-20.0,-720.0          # the spine's height in atlas mm (skull base to coccyx), with margin
secs=[];off=[0];parts=[]
def put(a):
    b=np.ascontiguousarray(a).tobytes();n=len(b);b+=b'\0'*((-n)%4);o=off[0];secs.append(b);off[0]+=len(b);return [o,n]
zr=[]
for fid,name,cl,zh in P:
    try:v,f=load(SRC+'FMA%d.stl'%fid)
    except FileNotFoundError:print('missing',fid,name);continue
    v=place(flip(v))
    fz=v[f].mean(1)
    keep=(fz[:,2]<ZT)&(fz[:,2]>ZB)
    pass
    if keep.sum()<20:continue
    f=f[keep];u,inv=np.unique(f,return_inverse=True);v=v[u];f=inv.reshape(-1,3)
    t=TGT[cl]
    if len(f)>t:v,f=fs.simplify(v.astype(np.float32),f.astype(np.int32),target_reduction=1-t/len(f))
    if len(v)>65535:print('too many verts',name);continue
    zr.append((name,round(float(v[:,2].min())),round(float(v[:,2].max()))))
    parts.append(dict(id='s%d'%fid,name=name,cl=cl,zh=zh,v=put(np.round(v*10).astype(np.int16)),f=put(np.asarray(f).astype(np.uint16))))
H=dict(parts=parts,src='BodyParts3D, (c) The Database Center for Life Science, licensed under CC Attribution-Share Alike 2.1 Japan')
hj=json.dumps(H,ensure_ascii=False,separators=(',',':')).encode();hj+=b' '*((-len(hj))%4)
blob=np.uint32(len(hj)).tobytes()+hj+b''.join(secs);g=gzip.compress(blob,9,mtime=0);open('../docs/spine-soft.wasm','wb').write(g)
for r in zr[:8]+[q for q in zr if q[0] in('Skin','Descending aorta','Inferior vena cava','Esophagus')]:print(r)
print('parts',len(parts),'raw %.2f MB gz %.2f MB'%(len(blob)/1e6,len(g)/1e6))
