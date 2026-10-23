# Nonspatial morphology measurements

```bash
python code/figure4/extract_from_reconstructions.py --verify --out generated/morphology
```

The extractor uses the released SRC stripe reconstructions and image-pixel torso
ROIs to measure 18 traits, identity means, standardized traits, phenotype distances
and pixel IoU. Verification compares the per-image traits with the frozen table.
The public module does not require or distribute geographic inputs.
