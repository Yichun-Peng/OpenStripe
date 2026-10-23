# Figure 1 workflow drawing

```bash
python reproduce.py --figure 1 --panel workflow
python code/figure1/render.py --panel all --output-dir generated/figure1
```

The renderer reads `data/figure1/workflow.json`, common-BW images S016/S066 and
their JSON annotations. It draws the four-agent workflow, reconstruction examples,
structural-data cards and seeded schematic clustering dots.

Use `--formats pdf,png,svg` and `--dpi 240` to choose exports. The optional
`--private-assets-dir PATH` accepts externally supplied `S016_annotated.png` and
`S066_annotated.png` illustration files.

`data/legacy_figshare_v1/data_A.npz` and `data_BC.npz` contain the previous
version's numerical feature arrays.
