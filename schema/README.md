# Images and annotations

S001–S100 represent 100 samples from 28 reference individuals, T01–T28.
Each sample has a common-BW reconstruction, a native-resolution SRC
reconstruction and a JSON annotation file.

- `data/images_unannotated/`: the common-BW annotation canvas used for clustering;
  11 unmarked outputs and 89 color-removal/inpainting outputs, with equal RGB channels.
- `data/images_unannotated_src/`: full-body reconstructions at their native size;
  SRC contains small compression-related RGB-channel differences.
- `data/annotations/`: B/F/I/C labels and coordinates, totaling 46/140/37/93 markers.
- `data/samples.csv`: identities, filenames, dimensions, marker counts and SHA256.

## Coordinate definitions

Annotation coordinates use the paired common-BW image frame. SRC has its own
native image frame, with morphology crops listed in `data/figure4/morphology_rois.csv`.

The origin is the top-left pixel center; x increases rightward and y downward.
Coordinates are zero-based continuous values. Normalized image coordinates are
`x_pixel / width` and `y_pixel / height`.

`bbox_xywh`, `area_pixel2` and `radius_pixel` describe parsed annotation components.
`body_bbox_xyxy` defines the foreground box with exclusive upper bounds.
Body coordinates are `(x-x0)/(x1-x0)` and `(y-y0)/(y1-y0)`, with unclipped values
for centroids outside that box.

Type codes are B (breakpoint), F (bifurcation), I (irregular structure) and
C (round structure). An absent type has an empty set of entries.

## Export and check

```bash
python scripts/export_annotations.py
python scripts/export_annotations.py --output-dir generated/annotations
python scripts/validate_release.py
```

The export command writes a flat 316-row CSV and `markers_primary316.json`.
`checksums_images_annotations.csv` contains file hashes for images, annotations
and the sample index. `annotation.schema.json` defines the JSON structure.
