import numpy as np,nibabel as nib,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
V=nib.load('out/vessel_mni.nii.gz').get_fdata()>0.5
lab,n=ndi.label(V); sz=np.bincount(lab.ravel()); keep=sz>=40; keep[0]=False; V=keep[lab]
np.save('out/vmask.npy',V)
sk=skeletonize(V); np.save('out/vskel.npy',sk)
print(V.sum(),sk.sum(),n)
O=np.array([-98,-134,-72.])
ext=lambda a,b:[O[a],O[a]+V.shape[a]*.5,O[b],O[b]+V.shape[b]*.5]
f,ax=plt.subplots(1,3,figsize=(30,12))
for i,(axis,(a,b),t) in enumerate([(2,(0,1),'axial MIP (x,y)'),(1,(0,2),'coronal MIP (x,z)'),(0,(1,2),'sagittal MIP (y,z)')]):
    m=V.max(axis); d=(np.argmax(V,axis) if axis!=2 else V.shape[2]-np.argmax(V[:,:,::-1],2))
    ax[i].imshow(np.where(m,d,np.nan).T,origin='lower',extent=ext(a,b),cmap='turbo');ax[i].set_title(t);ax[i].grid(True,alpha=.4)
    ax[i].set_xticks(np.arange(-100,101,10));ax[i].set_yticks(np.arange(-130,100,10));ax[i].tick_params(labelsize=7)
plt.savefig('test/vmip.png',dpi=55,bbox_inches='tight')
