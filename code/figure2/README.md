# Figure 2 drawing

```bash
python code/figure2/render.py --panel all --output-dir generated/figure2
python reproduce.py --figure 2 --panel a
python reproduce.py --figure 2 --panel b
python reproduce.py --figure 2 --panel c
```

`--panel all` produces the three panels and their combined figure. Exports use
PDF, SVG and PNG. Set `--formats pdf,png,svg`, `--dpi 240` and `--output-dir PATH`.

## Numerical panels a and b

`data/figure2/archived_plot_values.json` supplies summary means and displayed
error-bar lengths. The fields `stds_A`, `pck_stds` and `nme_stds` store those
display lengths. `quantitative_plot_values.csv` contains the same nine metric rows.
The renderer places bar heights and error-bar endpoints using these values.

## Reconstruction gallery c

Six paired rows display SRC and common-BW images for S001, S002, S003, S009,
S027 and S093. The first three samples share reference identity T01. Four
structural-data cards use S009, S027, S079 and S093, with B/F/I/C counts read
from their JSON files and printed beneath each image.

## External illustration mode

```bash
python code/figure2/render.py --panel c --private-assets-dir PATH --output-dir generated/figure2_external
```

This mode reads externally supplied camera/reconstruction pairs for A1, A2,
A3, B, C and D, named `A1_camera.png`, `A1_reconstruction.png`, and so on,
plus `A_topology.png` through `D_topology.png`. Its four endpoint/bifurcation
count pairs are in `data/figure2/private_illustration_counts.json`.
