import ants,numpy as np,nibabel as nib,sys,os,time
D='../data/raw/'
tpl=ants.image_read(D+'mouches/tofAverage.nii')
tpl1=ants.resample_image(tpl,(1,1,1),False,0)
for sid in sys.argv[1:]:
    t0=time.time();out='out/tb/%s'%sid
    if os.path.exists(out+'_mr_lab.nii.gz'):continue
    mr=ants.image_read(D+'topbrain/topcow_mr_%s_0000.nii'%sid);mr1=ants.resample_image(mr,(1,1,1),False,0)
    reg=ants.registration(fixed=tpl1,moving=mr1,type_of_transform='SyN',aff_metric='mattes',syn_metric='mattes')
    ml=ants.image_read(D+'topbrain/topcow_mr_%s.nii'%sid)
    w=ants.apply_transforms(fixed=tpl,moving=ml,transformlist=reg['fwdtransforms'],interpolator='genericLabel')
    nib.save(nib.Nifti1Image(w.numpy().astype(np.uint8),nib.load(D+'mouches/tofAverage.nii').affine),out+'_mr_lab.nii.gz')
    wm=ants.apply_transforms(fixed=tpl1,moving=mr1,transformlist=reg['fwdtransforms'])
    ants.image_write(wm,out+'_mr_warped.nii.gz')
    print(sid,'MR done',round(time.time()-t0),flush=True)
    ct=ants.image_read(D+'topbrain/topcow_ct_%s_0000.nii'%sid);ct1=ants.resample_image(ct,(1,1,1),False,0)
    ct1=ants.threshold_image(ct1,-100,3000,1,0)*ct1   # drop air
    r2=ants.registration(fixed=mr1,moving=ct1,type_of_transform='Affine',aff_metric='mattes',aff_sampling=48)
    cl=ants.image_read(D+'topbrain/topcow_ct_%s_v1.nii'%sid)
    w2=ants.apply_transforms(fixed=tpl,moving=cl,transformlist=reg['fwdtransforms']+r2['fwdtransforms'],interpolator='genericLabel')
    nib.save(nib.Nifti1Image(w2.numpy().astype(np.uint8),nib.load(D+'mouches/tofAverage.nii').affine),out+'_ct_lab.nii.gz')
    print(sid,'CT done',round(time.time()-t0),flush=True)
