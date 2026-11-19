import ants,numpy as np,nibabel as nib
from grid import res,O,S
import grid
R='../data/raw/'
H=R+'HQA_Cerebrovasculature/'
mra=ants.image_read(H+'2_CereVessSeg_CereVessPro/MRA_data/MRA/Subject0001-MRA_0000.nii.gz')
t1=ants.image_read(H+'1_preprocessing/imgs/T1/Subject0001/Subject0001-T1.nii.gz')
tpl=ants.image_read(R+'dsa/human/human_T1w.nii.gz')
r1=ants.registration(fixed=t1,moving=mra,type_of_transform='Rigid',aff_metric='mattes')
tplc=ants.resample_image(tpl,(1,1,1),False,0)
r2=ants.registration(fixed=tplc,moving=t1,type_of_transform='SyN')
tx=r2['fwdtransforms']+r1['fwdtransforms']
fov=ants.apply_transforms(fixed=tpl,moving=mra*0+1,transformlist=tx,interpolator='linear')
w=ants.apply_transforms(fixed=tpl,moving=mra,transformlist=tx,interpolator='linear')
seg=ants.image_read(H+'2_CereVessSeg_CereVessPro/MRA_data/MRA_GT/Subject0001-MRA.nii.gz')
ws=ants.apply_transforms(fixed=tpl,moving=seg.astype('float32'),transformlist=tx,interpolator='linear')
ants.image_write(w,'out/mra_mni.nii.gz');ants.image_write(fov,'out/mrafov_mni.nii.gz');ants.image_write(ws,'out/vessel_mni.nii.gz')
