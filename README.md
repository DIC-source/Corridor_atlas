# Corridor Atlas

Atlas-based neuroanatomy viewer for teaching and pre-operative rehearsal: an illustrated atlas plate / 3D model / long-pathway view on the left, MRI or MRA with outline annotations on the right, and a Lesion & corridor planning mode: place the lesion and the routes to it are ranked automatically (31 standard neurosurgical approaches, the best direct cortical entry and the two-point line), each with the reasons for it; the ranking method, risk-score factors and key references are in the Sources tab (indication, why this approach, what it risks, how it compares), a heuristic trajectory risk score, and the corridor drawn from scalp to lesion. Selecting a structure shows its afferent and efferent pathways (textbook) and the atlas tracts that run through it.

The site opens on a choice of two workspaces:

- **Brain atlas**: the brain viewer and corridor planning described above.
- **Spinal pathways**: C0 to the coccyx.
  - **Anatomy**: textbook cross-section plate, 3D, PAM50 T2 MRI and a CT-style rendering. Click any structure to name it.
  - **Cord tracts**: the cord section and 3D block with the textbook tract layout, segment by segment.
  - **Approach**: spine procedures named as in the literature (ACDF, TLIF, OLIF, TELD…). Each lists its corridor, visualisation and invasiveness. They are ranked for a lesion you place on the images, with inline axial and sagittal views along the route.

> **Teaching and rehearsal only.** Everything is built from population atlases (MNI ICBM152, PAM50) and a generic reference body (BodyParts3D), not from any patient. Real planning needs the patient's own imaging, DTI tractography, functional mapping and navigation.

## Run it

It is a static site: `docs/index.html` plus gzip-compressed data files decompressed in the browser:
- `atlas-data.wasm`: brain core.
- `atlas-extra.wasm`: brain T2, MRA and 3D meshes, streamed in after the core.
- `plan-data.wasm`: brain planning layers.
- `spine-data.wasm`: spine skeleton, cord, roots and vessels.
- `spine-mri.wasm`: PAM50.
- `spine-soft.wasm`: spine soft tissue.
- `spine-ws.wasm`: whole-spine population T2 template.

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
| `docs/atlas-data.wasm`, `docs/atlas-extra.wasm` | Brain: packed volumes, meshes, tracts, vessels, pathway geometry (core first, extra streamed after) |
| `docs/plan-data.wasm` | Brain planning layers: distance to vessels and sulci, critical-tract directions, entry candidates |
| `docs/spine-data.wasm` | Spine: BodyParts3D vertebrae, sacrum, ribs and discs placed on the atlas cord, with measured canal and body sizes per level |
| `docs/spine-mri.wasm` | Spine: PAM50 T2 template, cord/CSF/grey-matter masks, white-matter tract labels and level labels |
| `docs/spine-soft.wasm` | Spine: BodyParts3D muscles, great vessels, viscera and skin, with the same placement as the bones |
| `docs/spine-ws.wasm` | Spine: population T2 template of the whole column (56 adults, OpenNeuro ds005616), built by `spinewst2.py` |
| `pipeline/` | Python scripts that build the data files from the source atlases |
| `data/SOURCES.md` | Where each input comes from and its licence |
| `data/raw/` | Downloaded inputs (git-ignored; see SOURCES.md) |

## Rebuild the data

Inputs go in `data/raw/` (see `data/SOURCES.md`). From `pipeline/`:

```bash
pip install nibabel scipy scikit-image fast-simplification antspyx skan
python3 build.py      # composite label volume + T1/T2 backgrounds
python3 fixlab.py     # split CerebrA ventral DC at the PC–mammillary plane (hypothalamus / crus / midbrain), STN at DISTAL centroid, SN grown 1 mm
python3 cnnuclei.py   # schematic cranial nerve nuclei placed relative to the brainstem
python3 build2.py     # structure meshes + HCP-1065 tract voxels/polylines
python3 oloops.py     # optic radiation Meyer loop (ventral) and Baum loop (dorsal) as schematic bundles on template landmarks; the group-average bundle truncates Meyer's loop
python3 build3.py     # arterial-territory grouping helper (writes out/ter_raw.npy-based groups)
python3 build4.py 25  # population vessel atlases: MRA volumes + vessel surfaces (probability >= 25%)
python3 trace.py      # named artery centerlines (minimal-cost paths on the probability atlas)
python3 branches.py   # distal named branches: MCA M2 trunks + M3/M4 cortical arteries, ACA/PCA cortical and choroidal branches, AComm, ophthalmic, anterior spinal (sulcal landmarks from the parcellation, sulci.py, pulled onto the probability ridge)
python3 veins.py      # dural sinuses / deep veins placed from template geometry (torcula and straight sinus fitted to the template T1)
python3 build_named.py  # tube meshes + voxel masks for named vessels
python3 schematic.py  # schematic surgical vessels (Heubner, LSA, AChA, SCA, AICA, PICA, Labbé, Trolard, bridging veins...) and CN I, III-XII
python3 paths.py && python3 paths2.py   # long-pathway (CST / DCML / STT) geometry; cortical ends pinned inside the precentral (M1) and postcentral (S1) gyri
python3 connect.py    # join every branch to its parent at its usual origin (e.g. Heubner on proximal A2, PComm on P1, AChA distal to PComm); repeated passes so moved parents carry their children
python3 pack.py       # writes ../docs/atlas-data.wasm (core) and ../docs/atlas-extra.wasm (T2, MRA, 3D meshes)
python3 plandata.py   # planning layers (vessel/sulcus distance, critical-tract directions incl. the Meyer/Baum loops, entry candidates) -> plan-data.wasm

# spine (inputs: data/raw/bodyparts3d/stl/, data/raw/pam50/)
python3 spinebp.py    # vertebrae, sacrum, ribs, discs placed on the atlas cord (C1 canal centre), canal and body sizes per level -> spine-data.wasm
python3 spinemri.py   # PAM50 T2 template, cord/CSF/GM masks, tract atlas, levels -> spine-mri.wasm
python3 spinesoft.py  # muscles, great vessels, viscera, skin with the same transform as the bones -> spine-soft.wasm

# whole-spine population T2 (input: data/raw/wholespine/ws/ = release wholespine_v1, unpacked)
STAGE=straighten python3 spinewst2.py   # straighten along the canal, match vertebrae and discs, scale to CSF, mean -> out/wst2/tmpl0
STAGE=syn python3 spinewst2.py ../data/raw/wholespine/ws out/wst2 1   # ANTs SyN of each person to the mean (about 1.5 h on 2 cores)
python3 spinewst2geom.py               # per-level vertebral body front, for scaling to the reference skeleton
python3 spinewst2pack.py out/wst2 1    # -> spine-ws.wasm (template, coverage x body weight, tissue/CSF ratios, level brightness)
```

The spine MRI is three layers: PAM50 in the canal (C1–L1), the whole-spine population T2 in the centre ±25 mm (C1 to about L4–L5; the scans are a sagittal slab), and around them a synthetic T2 of the reference body whose tissue brightness is measured on that template. Only the synthetic part is not a scan.

The spine's roots, sympathetic trunk, small vessels, carotid sheath above C7, pharynx, larynx and tract blocks are schematic and are built in `index.html` from the measured vertebral levels. The Sources tab says which parts are schematic.

Intermediate files go to `pipeline/out/` (git-ignored). Viewer controls: mouse wheel pages slices, Ctrl + wheel or pinch zooms the plate and MRI, right- or Shift-drag pans, double-click fits. The pathway geometry (`paths.json`) is currently embedded in `index.html` as `const PW=...`; regenerate and paste if you change it.

## Data and licences

See `data/SOURCES.md`. Short version: ICBM152 (copyright notice required), CerebrA, THOMAS (CC0), ATAG, SUIT (CC BY-NC 3.0), HCP-1065 tractography atlas (CC BY-SA 4.0 + HCP acknowledgment), population vessel atlases of Mouches & Forkert 2019 (cite the paper), BodyParts3D (CC BY-SA 2.1 JP, attribution required; the spine meshes derived from it are share-alike), PAM50 (no licence file in its repository; cite De Leener et al. 2018), OpenNeuro ds005616 Whole-Spine (CC0; cite Molinier et al.). Check these before making the repository or a hosted copy public.

## Trajectory risk score

The score in Lesion & corridor is a teaching heuristic. The factors it checks come from published automated trajectory planners: vessel clearance of at least 3 mm, avoiding sulci, ventricles, path length and entry angle (Vakharia et al., Neurotherapeutics 2019, EpiNav; Liu et al., IEEE TBME 2014, DBS planning). Two factors follow general neurosurgical principles: the length of brain crossed before the lesion, and an approach-related cost (for example, translabyrinthine sacrifices hearing). Natural-corridor approaches skip the sulcus, entry-cortex, angle and transgression checks, and their vessel cost is capped. The entry search also offers the two-point method (Brown, Thompson & Spetzler, BNI Quarterly 1996). The weights, the thresholds, the approach costs and the low / moderate / high bands are this project's own and have not been validated. The viewer explains each factor in the Sources tab under "風險分數怎麼算、參考什麼", with the five key references for the ranking.

## Note on TopBrain

The TopBrain cases are kept as a reference dataset only (release `topbrain_v1`) and are **not used** in the packed data. The single-case MRA slabs did not register reliably to the whole-brain template, and the atlas deliberately shows generic (population) anatomy rather than one person's variants. The scripts that handled them (`tbreg.py`, `tbreg2.py`, `tbct.py`, `tbbuild.py`, `popbuild.py`, `assemble.py`) stay in `pipeline/` for reference and are not part of the build above. Keeping the release has no effect on the site: Pages serves only `docs/`.
