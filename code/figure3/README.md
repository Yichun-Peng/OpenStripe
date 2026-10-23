# Figure 3 clustering analyses

## Draw

```bash
python code/figure3/render.py --panel all
python reproduce.py --figure 3 --panel c2
python reproduce.py --figure 3 --panel d4
```

Panel labels are `a`, `b`, `c1`–`c4`, `d1`–`d4` and `e`; `c` and `d` draw grouped
panels. Use `--output-dir PATH`, `--formats pdf png svg`, `--dpi`, `--data-dir`
and `--image-dir` to configure drawing. The circular tree uses pattern thumbnails.

## Summaries, statistics and consensus

```bash
python code/figure3/reanalyse.py --refit-consensus
python code/figure3/render.py --data-dir generated/figure3_reanalysis --panel all
```

- a: 612 metric means across 17 inputs, nine methods and four metrics.
- b: equal-weight means for zero to four annotation classes; the zero-class input is BW.
- c1–c4: 112 paired gains and 16 means, adding each class to seven background subsets.
- d1–d4: metrics for 50 saved partitions; metric-wise OpenStripe–best-baseline
  differences and paired delete-one-reference-identity jackknife t tests, df = 27.
  The best baseline is reselected within each deletion.
- e: equal-weight consensus from 27 membership groups, average-linkage clustering
  and automatic K selection over 2–60. The stored display-order linkage supplies the ring layout.

`--bootstrap 1999` additionally runs fixed-partition identity bootstrap diagnostics
with seed 20261015. Displayed p values use the identity-jackknife calculation.
The reanalysis command copies the layout inputs into its output directory.

## Extract features

```bash
python -m pip install -r code/figure3/requirements-features.txt
python code/figure3/extract_from_public_inputs.py --assignment-distances
```

The extractor reads 100 common-BW PNGs and their JSON annotations. It computes
canvas/body coordinates, the 316-point type array, 128-dimensional typed shape
contexts, 432-dimensional local multiscale HOG, soft occupancy maps, moments,
pooled HOG and pairwise distances. `reference_features/` supplies optional
comparison arrays; `--skip-reference-comparison` selects feature extraction alone.

For DINOv2 ViT-S/14 inference, install `requirements-neural.txt` and supply the
checkpoint listed in `data/figure3/model_weights.json`:

```bash
python code/figure3/extract_from_public_inputs.py --assignment-distances --dino-weights PATH --output-dir generated/figure3_features_dino
```

The checkpoint hash is checked before inference. Body crops are padded to
518 × 518 pixels; 37 × 37 grids of 384-dimensional tokens are pooled around
marker positions at two bandwidths. Outputs include 768-dimensional local
descriptors, 32-dimensional whitened PCA features and typed pooling.
The model inventory lists identifiers, checkpoint versions and hashes for the
saved baseline configurations. Baseline-comparison statistics read their saved partitions.
