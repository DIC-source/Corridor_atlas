# Re-register each CTA to its own MRA by aligning the shared arterial annotations (labels 1-34), then reuse the MRA->MNI warp
import ants,numpy as np,nibabel as nib,sys,time
D='../data/raw/'
tpl=ants.image_read(D+'mouches/tofAverage.nii');tpl1=ants.resample_image(tpl,(1,1,1),False,0)
for sid in sys.argv[1:]:
    t0=time.time()
    mr=ants.image_read(D+'topbrain/topcow_mr_%s_0000.nii'%sid);mr1=ants.resample_image(mr,(1,1,1),False,0)
    reg=ants.registration(fixed=tpl1,moving=mr1,type_of_transform='SyN',aff_metric='mattes',syn_metric='mattes')
    ml=ants.image_read(D+'topbrain/topcow_mr_%s.nii'%sid);cl=ants.image_read(D+'topbrain/topcow_ct_%s_v1.nii'%sid)
    def vmask(img):
        a=img.numpy();m=((a>=1)&(a<=34)).astype('float32');i=ants.from_numpy(m,origin=img.origin,spacing=img.spacing,direction=img.direction)
        i=ants.resample_image(i,(1,1,1),False,0);return ants.smooth_image(i,1.5)
    fm,mm=vmask(ml),vmask(cl)
    best=None
    for init in ['Rigid','Affine']:
        r=ants.registration(fixed=fm,moving=mm,type_of_transform=init,aff_metric='meansquares',initial_transform=None if init=='Rigid' else best[1])
        best=(init,r['fwdtransforms'])
    w2=ants.apply_transforms(fixed=tpl,moving=cl,transformlist=reg['fwdtransforms']+best[1],interpolator='genericLabel')
    nib.save(nib.Nifti1Image(w2.numpy().astype(np.uint8),nib.load(D+'mouches/tofAverage.nii').affine),'out/tb/%s_ct_lab.nii.gz'%sid)
    # agreement check: Dice of the shared arteries after warping both
    w1=ants.apply_transforms(fixed=tpl1,moving=ml,transformlist=reg['fwdtransforms'],interpolator='genericLabel').numpy()
    w3=ants.apply_transforms(fixed=tpl1,moving=cl,transformlist=reg['fwdtransforms']+best[1],interpolator='genericLabel').numpy()
    a=(w1>=1)&(w1<=34);b=(w3>=1)&(w3<=34);dA=ndice=2*(ndi:=(a&b)).sum()/(a.sum()+b.sum())
    import scipy.ndimage as S;aa=S.binary_dilation(a,iterations=2);bb=S.binary_dilation(b,iterations=2)
    print(sid,'dice',round(float(dA),2),'dice@2mm',round(float(2*(aa&bb).sum()/(aa.sum()+bb.sum())),2),round(time.time()-t0),flush=True)
