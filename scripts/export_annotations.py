#!/usr/bin/env python3
"""Check the 100 sample/image/annotation pairs and export annotation CSV/JSON tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


TYPES = {"B": "breakpoint", "F": "bifurcation", "I": "irregular", "C": "round"}
FIELDS = ["sample_id", "identity_id", "annotation_id", "type", "marker_type",
          "x_pixel", "y_pixel", "x_normalized", "y_normalized",
          "body_x", "body_y", "canvas_width", "canvas_height",
          "body_bbox_x0", "body_bbox_y0", "body_bbox_x1", "body_bbox_y1",
          "area_pixel2", "radius_pixel", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Nonportable file reference: {relative}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_public(root):
    with (root / "data/samples.csv").open(encoding="utf-8", newline="") as fh:
        samples = list(csv.DictReader(fh))
    ids = [f"S{i:03d}" for i in range(1, 101)]
    if [s["sample_id"] for s in samples] != ids:
        raise ValueError("Expected the fixed S001--S100 cohort in sample order")
    records, points, counts = [], [], Counter()
    image_checks = []
    for row in samples:
        p = safe_file(root, row["annotation"])
        record = json.loads(p.read_text(encoding="utf-8"))
        assert record["sample_id"] == row["sample_id"]
        assert record["identity_id"] == row["identity_id"]
        assert record["source_version"] == "main_clustering_components316"
        image = record["image"]
        assert image["file"] == row["common_bw_image"]
        ip = safe_file(root, image["file"])
        with Image.open(ip) as im:
            assert im.size == (image["width"], image["height"])
            assert not im.info, "PNG metadata must be absent"
            rgb = np.asarray(im.convert("RGB"))
        assert np.array_equal(rgb[..., 0], rgb[..., 1])
        assert np.array_equal(rgb[..., 1], rgb[..., 2])
        assert sha(ip) == image["sha256"] == row["common_bw_sha256"]
        assert p.stem == ip.stem == row["sample_id"]
        x0, y0, x1, y1 = record["coordinate_system"]["body_bbox_xyxy"]
        w, h = image["width"], image["height"]
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        for a in record["annotations"]:
            t, x, y = a["type"], a["x_pixel"], a["y_pixel"]
            assert t in TYPES and a["marker_type"] == TYPES[t]
            assert 0 <= x < w and 0 <= y < h
            assert abs(x / w - a["x_normalized"]) < 1e-12
            assert abs(y / h - a["y_normalized"]) < 1e-12
            assert abs((x - x0) / (x1 - x0) - a["body_x"]) < 1e-12
            assert abs((y - y0) / (y1 - y0) - a["body_y"]) < 1e-12
            b = a["bbox_xywh"]
            points.append(dict(sample_id=row["sample_id"], identity_id=row["identity_id"],
                annotation_id=a["annotation_id"], type=t, marker_type=TYPES[t],
                x_pixel=x, y_pixel=y, x_normalized=a["x_normalized"],
                y_normalized=a["y_normalized"], body_x=a["body_x"], body_y=a["body_y"],
                canvas_width=w, canvas_height=h, body_bbox_x0=x0, body_bbox_y0=y0,
                body_bbox_x1=x1, body_bbox_y1=y1, area_pixel2=a["area_pixel2"],
                radius_pixel=a["radius_pixel"], bbox_x=b[0], bbox_y=b[1],
                bbox_w=b[2], bbox_h=b[3]))
            counts[t] += 1
        own_counts = Counter(a["type"] for a in record["annotations"])
        assert all(own_counts[t] == int(row[f"primary_{t}_count"]) for t in TYPES)
        sp = safe_file(root, row["src_image"])
        with Image.open(sp) as im:
            assert im.size == (int(row["src_width"]), int(row["src_height"]))
            assert not im.info, "SRC metadata must be absent"
            a = np.asarray(im.convert("RGB"), dtype=np.int16)
            # Archived monochrome reconstructions contain weak RGB compression
            # differences. Preserve pixels; do not recolor the experiment input.
            spread = a.max(axis=2) - a.min(axis=2)
            assert int(spread.max()) <= 30, "Unexpected colored annotation/image"
        assert sha(sp) == row["src_sha256"]
        image_checks.append(dict(sample_id=row["sample_id"],
            common_bw_sha256=sha(ip), src_sha256=sha(sp),
            annotation_sha256=sha(p), common_bw_strict_grayscale=True,
            src_max_rgb_channel_difference=int(spread.max())))
        records.append(record)
    assert counts == {"B": 46, "F": 140, "I": 37, "C": 93}
    assert len({s["identity_id"] for s in samples}) == 28
    return records, points, counts, image_checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output-dir", type=Path,
                    help="Optional destination for point CSV and original-format marker JSON")
    args = ap.parse_args()
    records, points, counts, checks = read_public(args.root.resolve())
    if args.output_dir:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        with (out / "annotation_points_primary316.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(points)
        legacy = []
        for r in records:
            markers = []
            for a in r["annotations"]:
                b = a["bbox_xywh"]
                markers.append(dict(marker_type=a["marker_type"], x=a["x_pixel"], y=a["y_pixel"],
                    area=a["area_pixel2"], radius=a["radius_pixel"],
                    bbox_x=b[0], bbox_y=b[1], bbox_w=b[2], bbox_h=b[3]))
            legacy.append(dict(sample_id=r["sample_id"], width=r["image"]["width"],
                               height=r["image"]["height"], markers=markers))
        (out / "markers_primary316.json").write_text(json.dumps(legacy, indent=2) + "\n")
        (out / "image_annotation_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", image_annotation_pairs=len(records),
        unmarked_reconstruction_images=2 * len(records), identities=28,
        primary_annotations=len(points), type_counts=dict(counts),
        coordinate_mapping="identity transform: primary annotations to common-BW",
        annotations_mapped_to_SRC=False, annotated_rasters_generated=False)))


if __name__ == "__main__":
    main()
