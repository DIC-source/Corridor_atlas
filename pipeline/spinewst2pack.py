"""Pack the whole-spine T2 template (spinewst2.py output) for the viewer -> ../docs/spine-ws.wasm

Layout: uint32 header length, JSON header, then
  t2 : uint8 [s][v][u]  T2 signal scaled so CSF = 200
  w  : uint8 [s][v][u]  blending weight: people covering the voxel (0 below 3, full from 8) x the template's body outline
Header: dims [nu, nv, ns], u0, v0 (mm of the first voxel; u = +right, v = +anterior from the canal centre), step (mm),
        lm {vertebra or disc name: s in mm from the first slice}, n {name: people}.
Usage: python3 spinewst2pack.py [out_dir] [round]
"""
import sys, json, gzip, numpy as np
O = sys.argv[1] if len(sys.argv) > 1 else 'out/wst2'
R = sys.argv[2] if len(sys.argv) > 2 else '2'
DST = sys.argv[3] if len(sys.argv) > 3 else '../docs/spine-ws.wasm'
g = json.load(open(O + '/grid.json'))
A = np.load(f'{O}/tmpl{R}.npy').astype(np.float32); C = np.load(f'{O}/count{R}.npy').astype(np.float32)
t2 = np.clip(A / 1000 * 200, 0, 255).round().astype(np.uint8)
# body mask per axial slice (template person's outline): blurred signal above air, holes filled, feathered 3 mm,
# so the viewer does not paint the template's air over the reference body (or its body over the reference's air)
import scipy.ndimage as ndi
B = ndi.gaussian_filter(A, (2, 2, 1)) > 60
B = np.stack([ndi.binary_fill_holes(ndi.binary_closing(B[:, :, k], iterations=3)) for k in range(B.shape[2])], -1)
Bf = np.clip(ndi.distance_transform_edt(B, sampling=(1, 1, 50)) / 3, 0, 1)
w = (np.clip((C - 2) / 6, 0, 1) * Bf * 255).round().astype(np.uint8)
t2b = np.ascontiguousarray(t2.transpose(2, 1, 0)).tobytes(); wb = np.ascontiguousarray(w.transpose(2, 1, 0)).tobytes()
# tissue signal over CSF, measured on the template itself; used for the synthetic sides so they continue the real image
import os
# posterior soft tissue split in two clusters (muscle, fat) per level; marrow = vertebral body centre; discs from the disc mid-planes
lm = g['landmarks']; vi = lambda v: int(round((v - g['v0']) / g['step'])); ui = lambda u: int(round((u - g['u0']) / g['step']))
mus, fat, mar, nuc, ann = [], [], [], [], []
for name, sv in lm.items():
    k = int(round(sv / g['step']))
    if '-' in name:
        d = A[ui(-4):ui(4), vi(10):vi(35), k]; nuc.append(np.percentile(d, 90)); ann.append(np.percentile(d, 25)); continue
    sl = A[:, :vi(-36), max(0, k - 3):k + 4]; x = sl[(sl > 60) & B[:, :vi(-36), max(0, k - 3):k + 4]]
    if len(x) > 200:
        c = np.percentile(x, [20, 80])
        for _ in range(20):
            l2 = np.abs(x[:, None] - c[None]).argmin(1); c = np.array([np.median(x[l2 == j]) for j in range(2)])
        mus.append(c[0]); fat.append(c[1])
    mar.append(np.median(A[ui(-2):ui(3), vi(8):vi(30), k]))
tis = {k: round(float(np.median(v)) / 1000, 3) for k, v in [('mus', mus), ('fat', fat), ('marrow', mar), ('nucleus', nuc), ('annulus', ann)]}
# overall soft-tissue brightness along the spine (relative to its median): applied to the synthetic sides so they follow the template's
# level-to-level changes (coil stations, marrow conversion) instead of one fixed brightness
lev = np.array([np.median(A[:, :vi(-20), k][B[:, :vi(-20), k] & (A[:, :vi(-20), k] > 60)]) if (B[:, :vi(-20), k] & (A[:, :vi(-20), k] > 60)).sum() > 50 else np.nan for k in range(A.shape[2])])
gd = np.isfinite(lev); lev = np.interp(np.arange(len(lev)), np.where(gd)[0], lev[gd]); lev = ndi.gaussian_filter1d(lev, 15 / g['step']); lev = np.clip(lev / np.median(lev), 0.5, 2)
print('tissue / CSF', tis)
# per-level vertebral body front (template mm from the canal centre), for scaling the template to the reference skeleton
geom = json.load(open(O + '/geom.json')) if os.path.exists(O + '/geom.json') else {}
h = {'lev': [round(float(x), 3) for x in lev], 'front': {k: v[0] for k, v in geom.items()}, 'tis': tis, 'dims': list(A.shape), 'u0': g['u0'], 'v0': g['v0'], 'step': g['step'], 'lm': g['landmarks'], 'n': g['n'],
     'src': 'OpenNeuro ds005616 Whole-Spine (CC0), 3D T2 1 mm; straightened, level-normalised, SyN-registered mean',
     't2': [0, len(t2b)], 'w': [len(t2b), len(wb)]}
hb = json.dumps(h).encode()
raw = np.uint32(len(hb)).tobytes() + hb + t2b + wb
open(DST, 'wb').write(gzip.compress(raw, 9, mtime=0))
print(DST, len(raw) / 1e6, 'MB raw')
