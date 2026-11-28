"""Fetch the OpenNeuro ds005616 whole-spine T2w images and labels (CC0),
crop everything above the top vertebra (no head/face), and pack them."""
import os, json, time, urllib.request, numpy as np, nibabel as nib, tarfile
BASE = 'https://s3.amazonaws.com/openneuro.org/ds005616/'
files = [l.strip() for l in open('.github/fetch/files.txt') if l.strip()]
os.makedirs('raw', exist_ok=True); os.makedirs('out/ws', exist_ok=True)
def get(p):
    d = os.path.join('raw', p)
    if os.path.exists(d): return d
    os.makedirs(os.path.dirname(d), exist_ok=True)
    for k in range(4):
        try:
            urllib.request.urlretrieve(BASE + p, d); return d
        except Exception as e:
            print('retry', p, e); time.sleep(5)
    return None
for p in files: get(p)
for p in ['participants.tsv', 'README', 'dataset_description.json']: get(p)
summ = {}
for t in [f for f in files if f.endswith('_T2w.nii.gz') and not f.startswith('derivatives')]:
    s = t.split('/')[0]
    try:
        im = nib.as_closest_canonical(nib.load('raw/' + t))
        segp = f'raw/derivatives/labels/{s}/anat/{s}_T2w_label-spine_dseg.nii.gz'
        if not os.path.exists(segp): summ[s] = {'skip': 'no spine_dseg'}; continue
        sg = nib.as_closest_canonical(nib.load(segp))
        if sg.shape != im.shape: summ[s] = {'skip': f'shape {sg.shape} vs {im.shape}'}; continue
        L = np.asarray(sg.dataobj).astype(np.int16)
        vz = np.where(np.isin(L, np.arange(11, 51)).any((0, 1)))[0]
        zs = abs(im.affine[2, 2])
        top = min(im.shape[2], int(vz.max() + 12 / zs) + 1)
        crop = lambda img: nib.Nifti1Image(np.asarray(img.dataobj)[:, :, :top], img.affine, img.header)
        nib.save(crop(im), f'out/ws/{s}_T2w.nii.gz')
        for suf in ['label-spine_dseg', 'label-SC_seg', 'label-canal_seg', 'label-CSF_seg', 'labels-disc-manual']:
            q = f'raw/derivatives/labels/{s}/anat/{s}_T2w_{suf}.nii.gz'
            if os.path.exists(q):
                g = nib.as_closest_canonical(nib.load(q))
                if g.shape == im.shape: nib.save(crop(g), f'out/ws/{s}_T2w_{suf}.nii.gz')
        u = np.unique(L); 
        summ[s] = {'shape': list(im.shape), 'zooms': [float(x) for x in im.header.get_zooms()[:3]],
                   'labels': [int(x) for x in u], 'crop_top': top}
    except Exception as e:
        summ[s] = {'skip': repr(e)}
    print(s, summ[s])
json.dump(summ, open('out/ws/summary.json', 'w'), indent=1)
for p in ['participants.tsv', 'README', 'dataset_description.json']:
    if os.path.exists('raw/' + p): os.replace('raw/' + p, 'out/ws/' + p)
with tarfile.open('out/wholespine_t2_cropped.tar', 'w') as tf: tf.add('out/ws', arcname='ws')
print(os.path.getsize('out/wholespine_t2_cropped.tar') / 1e6, 'MB')
