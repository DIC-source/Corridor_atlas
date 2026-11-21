# Corridor Atlas

Atlas-based neuroanatomy viewer for teaching and pre-operative rehearsal: an illustrated atlas plate / 3D model / long-pathway view on the left, MRI or MRA with outline annotations on the right, plus lesion-proximity and entry-to-target corridor checks.

> **Teaching and rehearsal only.** Everything is built from population atlases in MNI ICBM152 space, not from any patient. Real planning needs the patient's own MRI, DTI tractography, functional mapping and navigation.

## Run it

It is a static site: `docs/index.html` plus one data file, `docs/atlas-data.bin` (gzip-compressed binary, decompressed in the browser).

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000
```

Opening `index.html` straight from disk will not work, because browsers block `fetch()` of local files. Any static host works (GitHub Pages, Netlify, an intranet web server). three.js r128 and d3 7.8.5 load from cdnjs/jsDelivr.

## Layout

| Path | What it is |
| --- | --- |
| `docs/index.html` | The whole viewer (HTML, CSS, JS in one file) |
| `docs/atlas-data.bin` | Packed volumes, meshes, tracts, vessels, pathway geometry |
| `pipeline/` | Python scripts that build `atlas-data.bin` from the source atlases |
| `data/SOURCES.md` | Where each input comes from and its licence |
| `data/raw/` | Downloaded inputs (git-ignored; see SOURCES.md) |

## Rebuild the data

Inputs go in `data/raw/` (see `data/SOURCES.md`). From `pipeline/`:

```bash
pip install nibabel scipy scikit-image fast-simplification antspyx skan
python3 build.py      # composite label volume + T1/T2 backgrounds
python3 build2.py     # structure meshes + HCP-1065 tract voxels/polylines
python3 vreg2.py      # warp example TOF-MRA + vessel annotation to MNI (ANTs SyN)
python3 skel.py       # vessel mask / skeleton
python3 build3.py     # vessel meshes and grouping
python3 paths.py && python3 paths2.py   # long-pathway (CST / DCML / STT) geometry
python3 pack.py       # writes ../docs/atlas-data.bin
```

Intermediate files go to `pipeline/out/` (git-ignored). The pathway geometry (`paths.json`) is currently embedded in `index.html` as `const PW=...`; regenerate and paste if you change it.

## Data and licences

See `data/SOURCES.md`. Short version: ICBM152 (copyright notice required), CerebrA, THOMAS (CC0), ATAG, SUIT (CC BY-NC 3.0), HCP-1065 tractography atlas (CC BY-SA 4.0 + HCP acknowledgment), example TOF-MRA from HQA_Cerebrovasculature (Apache-2.0 repository; image provenance not stated). Check these before making the repository or a hosted copy public.
