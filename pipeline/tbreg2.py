# Second stage: align each warped case's vessels to the population vessel-probability map (vessel-to-vessel registration)
import ants,numpy as np,nibabel as nib,sys
D='../data/raw/'
Pn=nib.load(D+'mouches/vesselProbabilities.nii');tpl=ants.image_read(D+'mouches/vesselProbabilities.nii')
fix=ants.smooth_image(ants.resample_image(tpl,(1,1,1),False,0)/100.0,1.0)
for sid in sys.argv[1:]:
    ml=ants.image_read(f'out/tb/{sid}_mr_lab.nii.gz');ml=ants.copy_image_info(tpl,ml)
    a=ml.numpy();big=np.isin(a,[1,2,3,4,5,6,7,8,9,10,11,12,17,19,23,24,35,36]).astype('float32')   # large, consistent arteries drive the fit
    mov=ants.smooth_image(ants.resample_image(ants.from_numpy(big,origin=tpl.origin,spacing=tpl.spacing,direction=tpl.direction),(1,1,1),False,0),1.0)
    reg=ants.registration(fixed=fix,moving=mov,type_of_transform='SyN',aff_metric='meansquares',syn_metric='meansquares',reg_iterations=(60,40,20),flow_sigma=4,total_sigma=1)
    P=Pn.get_fdata(dtype=np.float32);m0=(a>=1)&(a<=34)
    for kind in ['mr','ct']:
        L=ants.image_read(f'out/tb/{sid}_{kind}_lab.nii.gz');L=ants.copy_image_info(tpl,L)
        w=ants.apply_transforms(fixed=tpl,moving=L,transformlist=reg['fwdtransforms'],interpolator='genericLabel').numpy().astype(np.uint8)
        nib.save(nib.Nifti1Image(w,Pn.affine),f'out/tb/{sid}_{kind}_lab2.nii.gz')
        if kind=='mr':
            m1=(w>=1)&(w<=34);bg=np.isin(w,[1,4,5,6,7,23,24,35,36])
            print(sid,'hit before',round(float((P[m0]>=10).mean()),2),'after',round(float((P[m1]>=10).mean()),2),'large',round(float((P[bg]>=10).mean()),2),flush=True)
