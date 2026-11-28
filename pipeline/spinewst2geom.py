"""Per-level geometry of the whole-spine template, from the subjects' labels (straightened space, majority vote):
vertebral body front, posterior-element centroid and pedicle width at each vertebra, in template mm (u right, v anterior).
The viewer scales the template to the reference skeleton level by level with these (spinewst2pack.py puts them in the header).
Usage: python3 spinewst2geom.py [out_dir]"""
import sys, glob, json, numpy as np
O = sys.argv[1] if len(sys.argv) > 1 else 'out/wst2'
g = json.load(open(O + '/grid.json')); u0, v0, st = g['u0'], g['v0'], g['step']
VERT = {41 - i: n for i, n in enumerate(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'] + [f'T{k}' for k in range(1, 13)] + [f'L{k}' for k in range(1, 6)] + ['S1'])}
files = sorted(glob.glob(O + '/sub-*_labs.npz'))
geo = {}
for lab, name in VERT.items():
    if name not in g['landmarks']: continue
    s = int(round(g['landmarks'][name] / st)); fr, po, la = [], [], []
    for f in files:
        L = np.load(f)['lab']
        if s >= L.shape[2]: continue
        sl = L[:, :, max(0, s - 2):s + 3] == lab
        if sl.sum() < 50: continue
        uu, vv, _ = np.nonzero(sl); u = uu * st + u0; v = vv * st + v0
        mid = np.abs(u) <= 3
        if mid.sum() < 10: continue
        fr.append(np.percentile(v[mid], 98))                          # body front
        post = mid & (v < -6)
        if post.sum() > 5: po.append(v[post].mean())                  # posterior elements (lamina + spinous), centroid
        ped = (np.abs(v) <= 2)
        if ped.sum() > 5: la.append(np.percentile(np.abs(u[ped]), 95))  # pedicle outer edge at the canal centre
    if len(fr) >= 3:
        geo[name] = [round(float(np.median(fr)), 1), round(float(np.median(po)), 1) if po else None, round(float(np.median(la)), 1) if la else None, len(fr)]
json.dump(geo, open(O + '/geom.json', 'w'), indent=0)
for k, v in geo.items(): print(k, v)
