# Figure 3 data dictionary

All sample-level rows use the anonymous sample IDs S001–S100. Identity labels T01–T28 are reference identities. Tree cluster aliases T01–T29 denote the 29 predicted groups C01–C29; reference identities use a separate T01–T28 mapping.

| File | Content | Panels |
|---|---|---|
| `ablation_all_runs_recomputed.csv` | Per-run scores with input condition, clustering method and random seed | a, b, c1–c4 |
| `ablation_17x9_four_metrics.csv` | 17 × 9 grid of means and run standard deviations | a |
| `method_display_order.csv` | Fixed nine-column order | a |
| `category_count_equal_weight_means.csv` | Consensus means over the 1, 4, 6, 4 and 1 subsets with 0–4 annotation types | b |
| `annotation_28_paired_additions.csv` | 28 subset additions × four score changes | c1–c4 |
| `annotation_contribution_means.csv` | Mean paired gain for each annotation type | c1–c4 |
| `baseline_all_50_conditions_recomputed.csv` | Four metrics for 49 image-only configurations and OpenStripe | d1–d4 |
| `paired_comparison_inputs.npz` | `reference` (100,), `predictions` (50,100), `conditions` (50,); zero-based labels; noise already encoded as separate singleton groups | d1–d4 |
| `paired_identity_jackknife.npy` | 28 deleted-reference-identity × four metric differences | d1–d4 |
| `paired_comparison.json` | Metric-specific best-baseline effects, standard errors and raw p values | d1–d4 |
| `reference_identity_labels.csv` | Reference sample order and identity mapping | d1–d4, e |
| `tree_sample_labels.csv` | Leaf order, saved automatic partition aliases, matched identity and ten mismatch flags | e |
| `automatic_linkage_display_order.npy` | 99 × 4 SciPy linkage matrix in deterministic blind sample order | e |
| `BFIC_automatic_partition.npz` | `auto_K` and archived `fixed_K29` partitions | e |
| `BFIC_consensus_distance.npy` | 100 × 100 source-balanced consensus distance | e |
| `BFIC_consensus_source_partitions.npz` | 400 anonymous source membership vectors; the JSON specifies which enter the 27 equal-weight groups | e / backend reanalysis |
| `consensus_sources.json` | Selected source-family/method groups and conditions | e / backend reanalysis |

`SRC` denotes the full-body stripe reconstruction; `BW` denotes the unannotated common stripe base. `B`, `F`, `I`, and `C` indicate breakpoint, bifurcation, irregular and round structure annotations. The input tables retain historical condition identifiers to preserve computational correspondence. All metrics use every image, and each noise image is a separate singleton scoring cluster.

Panel e uses reconstruction thumbnails at `../images_unannotated/Sxxx.png`.

`reference_features/` contains anonymous numerical feature arrays and selected original distance matrices for verification of `extract_from_public_inputs.py`. They are optional comparison targets. The extraction recomputes features from `data/images_unannotated/` and `data/annotations/` before reading these reference arrays.

`model_weights.json` lists model download URLs, versions and hashes. The optional extraction entry point accepts a DINOv2 ViT-S/14 checkpoint through `--dino-weights PATH`. Saved baseline partitions supply the metric comparison inputs.
