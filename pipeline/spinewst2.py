"""Population whole-spine T2 template from the OpenNeuro ds005616 "Whole-Spine" dataset (CC0).

Input : data/raw/wholespine/ws/  sub-*_T2w.nii.gz + *_T2w_label-spine_dseg / -canal_seg / -SC_seg / -CSF_seg,
        participants.tsv  (the release mirror wholespine_v1, already cropped below the skull)
Output: pipeline/out/wst2/ (template, labels, landmark table); pack with spinewst2pack.py

Method (same idea as PAM50 and the MNI templates, extended to the whole column):
  1. straighten: canal centreline per subject, planes perpendicular to it (so lordosis/kyphosis do not blur the mean)
  2. level-normalise: vertebra and disc centroids are landmarks; template position of each = mean over subjects
     (relative to C1); each subject is resampled with a piecewise-linear map template -> subject arc length,
     only within 20 mm of its own end landmarks (its true coverage)
  3. intensity: divide by the CSF signal along the cord (removes scanner gain and coil-station steps)
  4. levels listed as stenotic in participants.tsv are left out of the mean for that subject (+/- 12 mm)
  5. average -> A; then rounds of ANTs SyN (each subject to the current mean, masked to its coverage) -> B, C
  6. labels warped with the last transforms and voted per voxel (vertebrae, discs, canal, cord, CSF)
Usage: python3 spinewst2.py [input_dir] [out_dir] [rounds]
"""
import sys, os, glob, json, csv, time, numpy as np, nibabel as nib, scipy.ndimage as ndi
from scipy.interpolate import make_smoothing_spline

D = sys.argv[1] if len(sys.argv) > 1 else '../data/raw/wholespine/ws'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'out/wst2'
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
STAGE = os.environ.get('STAGE', 'all')            # 'straighten' | 'syn' | 'labels' | 'all'
os.makedirs(OUT, exist_ok=True)
HW, AP0, AP1, ST = 26, -95, 60, 1.0              # mm: half width, posterior..anterior (rel. canal), step
VERT = {41 - i: n for i, n in enumerate(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'] + [f'T{k}' for k in range(1, 13)]
                                          + [f'L{k}' for k in range(1, 6)] + ['S1'])}      # 41 = C1 ... 17 = S1
DNAMES = ['C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-T1'] + [f'T{k}-T{k+1}' for k in range(1, 12)] + \
         ['T12-L1'] + [f'L{k}-L{k+1}' for k in range(1, 5)] + ['L5-S1']
DISC = {224 - i: n for i, n in enumerate(DNAMES)}                                       # 224 = C2-C3 ... 202 = L5-S1
DLAB = {n: l for l, n in DISC.items()}
LMK = set(VERT) | set(DISC)

def load(p):
    im = nib.as_closest_canonical(nib.load(p)); return im, np.asarray(im.dataobj)

def centreline(canal, aff):
    pts = [(*np.argwhere(canal[:, :, z]).mean(0), z) for z in np.where(canal.any((0, 1)))[0] if canal[:, :, z].sum() >= 8]
    w = (aff[:3, :3] @ np.array(pts, float).T).T + aff[:3, 3]
    w = w[np.argsort(w[:, 2])]; z = w[:, 2]
    zz = np.arange(z.min(), z.max(), 0.5)
    c = np.stack([make_smoothing_spline(z, w[:, 0], lam=2e3)(zz), make_smoothing_spline(z, w[:, 1], lam=2e3)(zz), zz], 1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    return c, s

def frames(c):
    t = np.gradient(c, axis=0); t /= np.linalg.norm(t, axis=1)[:, None]
    ex = np.array([1., 0, 0])[None] - t[:, :1] * t; ex /= np.linalg.norm(ex, axis=1)[:, None]
    return t, ex, np.cross(t, ex)                    # ex: +x (RAS right), ey: anterior

subs = sorted({os.path.basename(f).split('_')[0] for f in glob.glob(D + '/sub-*_T2w.nii.gz')})
part = {r['participant_id']: r for r in csv.DictReader(open(D + '/participants.tsv'), delimiter='\t')}

if STAGE in ('straighten', 'all'):
    info = {}
    for sb in subs:
        im, I = load(f'{D}/{sb}_T2w.nii.gz'); L = load(f'{D}/{sb}_T2w_label-spine_dseg.nii.gz')[1].astype(np.int16)
        canal = load(f'{D}/{sb}_T2w_label-canal_seg.nii.gz')[1] > 0
        c, s = centreline(canal, im.affine)
        lm = {}
        for lab in np.unique(L):
            if int(lab) not in LMK: continue
            p = np.argwhere(L == lab)
            if len(p) < 30: continue
            w = im.affine[:3, :3] @ p.mean(0) + im.affine[:3, 3]
            lm[int(lab)] = float(s[np.argmin(np.linalg.norm(c - w, axis=1))])
        info[sb] = dict(c=c, s=s, lm=lm)
        print(sb, 'length %.0f mm' % s[-1], 'landmarks', len(lm), flush=True)
    anchor = 41
    labs = sorted({l for v in info.values() for l in v['lm']})
    T = {l: float(np.mean([v['lm'][l] - v['lm'][anchor] for v in info.values() if l in v['lm']])) for l in labs}
    N = {l: sum(l in v['lm'] for v in info.values()) for l in labs}
    labs = [l for l in labs if N[l] >= 3]
    order = sorted(labs, key=lambda l: T[l]); ts = np.array([T[l] for l in order])
    S0, S1 = ts[0] - 14, ts[-1] + 14
    sgrid = np.arange(S0, S1, ST); ug = np.arange(-HW, HW + ST, ST); vg = np.arange(AP0, AP1 + ST, ST)
    json.dump({'landmarks': {(VERT.get(l) or DISC.get(l)): T[l] - S0 for l in order}, 'n': {(VERT.get(l) or DISC.get(l)): N[l] for l in order},
               'step': ST, 'u0': -HW, 'v0': AP0, 'shape': [len(ug), len(vg), len(sgrid)], 'subjects': subs,
               'axes': 'u = +right (RAS x) mm, v = +anterior mm from the canal centre, s = arc length from the bottom (mm)'},
              open(OUT + '/grid.json', 'w'), indent=1)
    acc = np.zeros((len(ug), len(vg), len(sgrid)), np.float32); cnt = np.zeros_like(acc)
    for sb, v in info.items():
        im, I = load(f'{D}/{sb}_T2w.nii.gz'); I = I.astype(np.float32)
        L = load(f'{D}/{sb}_T2w_label-spine_dseg.nii.gz')[1]
        own = [l for l in order if l in v['lm']]
        tl = np.array([T[l] for l in own]); sl = np.array([v['lm'][l] for l in own])
        sub_s = np.interp(sgrid, tl, sl)          # inside the landmark range
        a0 = (sl[1] - sl[0]) / (tl[1] - tl[0]); a1 = (sl[-1] - sl[-2]) / (tl[-1] - tl[-2])
        sub_s = np.where(sgrid < tl[0], sl[0] + (sgrid - tl[0]) * a0, sub_s)
        sub_s = np.where(sgrid > tl[-1], sl[-1] + (sgrid - tl[-1]) * a1, sub_s)
        ok = (sgrid > tl[0] - 20) & (sgrid < tl[-1] + 20) & (sub_s >= v['s'][0]) & (sub_s <= v['s'][-1])
        # stenotic levels (participants.tsv) left out for this subject
        for nm in (part.get(sb, {}).get('stenosis') or 'n/a').split(','):
            nm = nm.strip()
            if nm in DLAB and DLAB[nm] in T: ok &= np.abs(sgrid - T[DLAB[nm]]) > 12
        t_, ex, ey = frames(v['c'])
        k = np.clip(np.interp(sub_s, v['s'], np.arange(len(v['s']))), 0, len(v['s']) - 1).round().astype(int)
        P = v['c'][k][None, None] + ug[:, None, None, None] * ex[k][None, None] + vg[None, :, None, None] * ey[k][None, None]
        inv = np.linalg.inv(im.affine); co = np.moveaxis(P @ inv[:3, :3].T + inv[:3, 3], -1, 0)
        val = ndi.map_coordinates(I, co, order=1, cval=np.nan)
        val[:, :, ~ok] = np.nan
        labs_s = {}
        for nm, key in [('spine_dseg', 'lab'), ('canal_seg', 'canal'), ('SC_seg', 'cord'), ('CSF_seg', 'csf')]:
            p = f'{D}/{sb}_T2w_label-{nm}.nii.gz'
            g = (load(p)[1] if os.path.exists(p) else np.zeros(I.shape, np.uint8)).astype(np.float32)
            labs_s[key] = ndi.map_coordinates(g, co, order=0, cval=0).astype(np.uint8)
        ref = np.array([np.nanmedian(val[:, :, j][labs_s['csf'][:, :, j] > 0]) if (labs_s['csf'][:, :, j] > 0).sum() > 4 else np.nan
                        for j in range(len(sgrid))])
        g = np.isfinite(ref)
        if g.sum() < 20: print('skip (no CSF)', sb); continue
        ref = ndi.gaussian_filter1d(np.interp(np.arange(len(ref)), np.where(g)[0], ref[g]), 6)
        val = val / ref[None, None] * 1000
        np.save(f'{OUT}/{sb}_str.npy', np.nan_to_num(val, nan=0).astype(np.float16))
        np.save(f'{OUT}/{sb}_cov.npy', np.isfinite(val))
        np.savez_compressed(f'{OUT}/{sb}_labs.npz', **labs_s)
        m = np.isfinite(val); acc[m] += val[m]; cnt[m] += 1
        print('straightened', sb, flush=True)
    np.save(OUT + '/tmpl0.npy', acc / np.maximum(cnt, 1)); np.save(OUT + '/count0.npy', cnt.astype(np.uint8))

if STAGE in ('syn', 'all'):
    import ants
    done = sorted(f[len(OUT) + 1:-8] for f in glob.glob(OUT + '/sub-*_str.npy'))
    start = int(os.environ.get('ROUND0', 1))
    for rd in range(start, ROUNDS + 1):
        A = np.load(f'{OUT}/tmpl{rd-1}.npy').astype(np.float32); fx = ants.from_numpy(A)
        acc = np.zeros_like(A); n = np.zeros_like(A)
        for sb in done:
            t = time.time(); m = np.load(f'{OUT}/{sb}_str.npy').astype(np.float32); cov = np.load(f'{OUT}/{sb}_cov.npy')
            mv = ants.from_numpy(m); mk = ants.from_numpy(cov.astype(np.float32))
            r = ants.registration(fx, mv, type_of_transform='SyNOnly', reg_iterations=(40, 20, 0), syn_metric='CC', syn_sampling=2,
                                  flow_sigma=3, total_sigma=0, moving_mask=mk, mask_all_stages=True)
            w = r['warpedmovout'].numpy()
            wm = ants.apply_transforms(fx, mk, r['fwdtransforms']).numpy() > 0.5
            acc[wm] += w[wm]; n[wm] += 1
            if rd == ROUNDS:
                keep = []
                for i, f in enumerate(r['fwdtransforms']):
                    d = f'{OUT}/{sb}_xf{i}' + ('.nii.gz' if f.endswith('.nii.gz') else os.path.splitext(f)[1])
                    os.replace(f, d); keep.append(d)
                json.dump(keep, open(f'{OUT}/{sb}_xf.json', 'w'))
            print(f'round {rd}', sb, '%.0fs' % (time.time() - t), flush=True)
        np.save(f'{OUT}/tmpl{rd}.npy', acc / np.maximum(n, 1)); np.save(f'{OUT}/count{rd}.npy', n.astype(np.uint8))

if STAGE in ('labels', 'all'):
    import ants
    R = ROUNDS; A = np.load(f'{OUT}/tmpl{R}.npy').astype(np.float32); fx = ants.from_numpy(A)
    done = sorted(f[len(OUT) + 1:-8] for f in glob.glob(OUT + '/sub-*_str.npy'))
    vote = {}                                    # label value -> accumulated hits
    cord = np.zeros_like(A); canal = np.zeros_like(A); csf = np.zeros_like(A); nn = np.zeros_like(A)
    for sb in done:
        if not os.path.exists(f'{OUT}/{sb}_xf.json'): continue
        xf = json.load(open(f'{OUT}/{sb}_xf.json'))
        Lz = np.load(f'{OUT}/{sb}_labs.npz'); cov = np.load(f'{OUT}/{sb}_cov.npy').astype(np.float32)
        W = lambda a, interp='nearestNeighbor': ants.apply_transforms(fx, ants.from_numpy(a.astype(np.float32)), xf, interpolator=interp).numpy()
        wc = W(cov, 'linear') > 0.5; nn += wc
        lab = W(Lz['lab'])
        for l in np.unique(lab):
            l = int(l)
            if l in LMK: vote.setdefault(l, np.zeros(A.shape, np.uint8)); vote[l] += (lab == l) & wc
        cord += W(Lz['cord'], 'linear') * wc; canal += W(Lz['canal'], 'linear') * wc; csf += W(Lz['csf'], 'linear') * wc
        print('labels', sb, flush=True)
    keys = sorted(vote); stack = np.stack([vote[k] for k in keys], -1).astype(np.float32)
    best = stack.argmax(-1); frac = stack.max(-1) / np.maximum(nn, 1)
    L = np.where(frac >= 0.5, np.array(keys)[best], 0).astype(np.uint8)
    np.save(OUT + '/labels.npy', L)
    np.save(OUT + '/prob.npy', np.stack([cord, canal, csf], -1) / np.maximum(nn, 1)[..., None])
    np.save(OUT + '/count_lab.npy', nn.astype(np.uint8))
    print('label set', [VERT.get(k) or DISC.get(k) for k in np.unique(L) if k])
