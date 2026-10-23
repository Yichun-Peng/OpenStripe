# OpenStripe

Data and reproduction code for **OpenStripe supports open-set identification
and spatial stripe profiling in wild Amur tigers**.

## Data

The dataset contains 100 samples from 28 reference individuals (S001–S100;
T01–T28), with two unannotated reconstruction versions per sample.

| Path | Content and use |
|---|---|
| `data/images_unannotated/` | 100 common-BW reconstructions; clustering and structural annotation coordinates |
| `data/images_unannotated_src/` | 100 native-resolution SRC reconstructions; morphology extraction |
| `data/annotations/` | One JSON per sample; B/F/I/C types, coordinates and image pairing |
| `data/samples.csv` | Sample identities, filenames, dimensions, marker counts and hashes |
| `data/figure1/`–`data/figure3/` | Nonspatial figure inputs, clustering memberships and resampling inputs |
| `data/figure4/` | Nonspatial morphology measurements from reconstructed stripes |
| `data/legacy_figshare_v1/` | `data_A.npz` and `data_BC.npz` |

The annotation set contains 316 markers: 46 breakpoints (B), 140 bifurcations
(F), 37 irregular structures (I) and 93 round structures (C).

Common-BW uses the structural-annotation canvas: 11 images are unmarked outputs
and 89 are color-removal/inpainting outputs. Most common-BW images are 1024 pixels
wide. SRC retains its native reconstruction size: 98 images are 3264 × 2448 pixels
and two are 3200 × 2400 pixels. The two versions have separate image frames.
JSON coordinates use the paired common-BW frame. Morphology ROIs use SRC pixels.
SRC contains small RGB-channel differences from image compression.

## Install and draw

Use Python 3.10 or later. The dependency versions are listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --list
python reproduce.py --figure 3 --panel c2
python reproduce.py --all
```

The dispatcher draws 15 nonspatial entries: the Figure 1 workflow, Figure 2a–c,
and Figure 3a/b/c1–c4/d1–d4/e. It also creates combined Figures 2–3.
Use `--output-dir PATH`, `--formats png` and `--dpi 150` to control exports.
`panel_manifest.csv` lists each entry's input files and command.

## Figure inputs and operations

| Figure | Inputs and operations |
|---|---|
| 1 | Workflow schematic with reconstruction examples, JSON cards and seeded schematic clustering dots |
| 2a–b | Bar plots from archived summary means and displayed error-bar lengths |
| 2c | Paired common-BW/SRC reconstruction gallery and annotation-count cards |
| 3a–e | Clustering summaries, marginal gains, partition-based metrics, identity-jackknife tests and consensus refitting; pattern thumbnails in the tree |

Figure 3 baseline comparisons use saved prediction partitions. The feature
extractor supports geometry, typed contexts, HOG and optional DINOv2 inference.
See `code/figure3/README.md` for model checkpoints and extraction options.

Original camera-trap images and all geospatial data, including coordinates,
locations, terrain, maps and camera-to-record linkages, are not publicly released
owing to geopolitical sensitivities and conservation concerns for Amur tigers
and other endangered species.

## Recompute

```bash
python scripts/export_annotations.py --output-dir generated/annotations
python code/figure3/reanalyse.py --refit-consensus --output-dir generated/figure3_reanalysis
python code/figure3/render.py --data-dir generated/figure3_reanalysis --panel all
python -m pip install -r code/figure3/requirements-features.txt
python code/figure3/extract_from_public_inputs.py --assignment-distances
python code/figure4/extract_from_reconstructions.py --verify --out generated/figure4_extracted
```

DINOv2 extraction uses the optional dependencies in
`code/figure3/requirements-neural.txt` and a checkpoint supplied with
`--dino-weights PATH`. Checkpoint identifiers and hashes are listed in
`data/figure3/model_weights.json`.


## File checks

```bash
python scripts/validate_release.py
```

This checks image/JSON pairing, cohort counts, coordinate conventions, panel
inputs, source syntax and `SHA256SUMS.txt`.

## Licenses

Code: MIT (`LICENSE`, `LICENSE-CODE`). Data: CC BY 4.0 (`LICENSE-DATA`).
Legacy arrays: [OpenStripe Figshare v1](https://doi.org/10.6084/m9.figshare.32149882.v1).
The quantile-regression solver retains the statsmodels BSD attribution in
`code/figure4/STATSMODELS_LICENSE.txt`.
