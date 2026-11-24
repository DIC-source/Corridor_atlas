# Data sources

| Input (place under `data/raw/`) | Source | Licence / terms |
| --- | --- | --- |
| `dsa/human/human_T1w.nii.gz`, `human_T2w.nii.gz` | ICBM 2009b nonlinear symmetric template, via github.com/frankyeh/DSI-Studio-atlas | © 1993–2009 Louis Collins, McConnell Brain Imaging Centre, MNI, McGill University; permissive, keep copyright notice |
| `dsa/human/CerebrA.nii.gz` | CerebrA (Manera et al., 2020, Sci Data), same repository | Open; cite the paper |
| `dsa/human/THOMAS.nii.gz` | THOMAS thalamic nuclei (Su et al., 2019) | CC0 |
| `dsa/human/ATAG_basal_ganglia.nii.gz` | ATAG (Keuken et al., 2014) | Licence not confirmed — check before public release |
| `dsa/human/Cerebellum-SUIT.nii.gz` | SUIT (Diedrichsen et al., 2009) | CC BY-NC 3.0 (non-commercial) |
| `dsa/human/FreeSurferSeg.nii.gz` | FreeSurfer segmentation of the template, same repository | Used only for a brain mask |
| `tract/human.tt.gz`, `tract/human.txt` | HCP-1065 population tractography atlas (Yeh, 2022, Nat Commun), github.com/data-others/tract | CC BY-SA 4.0. Acknowledge: Data were provided in part by the Human Connectome Project, WU-Minn Consortium (PIs David Van Essen and Kamil Ugurbil; 1U54MH091657) |
| `HQA_Cerebrovasculature/...` (Subject0001 T1, TOF-MRA, vessel annotation) | github.com/YingChen7/HQA_Cerebrovasculature | Repository is Apache-2.0; original image source not stated — check before public release |
| `Acute-stroke_Detection_Segmentation/data/template/ArterialAtlas_padding.nii.gz` | Arterial territory atlas (Liu et al., 2023), github.com/Chin-Fu-Liu/Acute-stroke_Detection_Segmentation | CC BY-SA 4.0; used only to group vessel branches |

| `mouches/tofAverage.nii`, `vesselProbabilities.nii`, `vesselRadius.nii` | Mouches P, Forkert ND. A statistical atlas of cerebral arteries generated using multi-center MRA datasets from healthy subjects. Sci Data 2019;6:29 (Calgary MIPLab). Mirrored as release assets: github.com/DIC-source/Corridor_atlas/releases/tag/mri_data_v1 | Check the licence on the original figshare record; cite the paper |

The single-subject HQA example MRA is no longer used in the packed data (replaced by the population atlases).

| `topbrain/topcow_{mr,ct}_{005..010}*.nii` + label maps | TopBrain challenge data release v3 (Zenodo 10.5281/zenodo.16623495), mirrored at github.com/DIC-source/Corridor_atlas/releases/tag/topbrain_v1 | Open use, attribution required; commercial use requires the data owner's permission. Cite Yang et al., medRxiv 2026 (TopBrain) and NEJM AI 2026 (TopCoW) |
