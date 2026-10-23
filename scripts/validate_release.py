#!/usr/bin/env python3
"""Check file hashes, cohort counts, coordinate conventions, panel inputs and Python syntax."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_annotations import read_public


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def distributed(path):
    rel = path.relative_to(ROOT)
    return not any(p in {"build", "generated", "__pycache__", ".venv", ".git"} for p in rel.parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-checksums", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    records, points, counts, checks = read_public(ROOT)
    samples = list(csv.DictReader((ROOT / "data/samples.csv").open()))
    for sample in samples:
        assert sha(ROOT / sample["annotation"]) == sample["annotation_sha256"]
    expected_images = {
        f"data/{folder}/S{i:03d}.png"
        for folder in ["images_unannotated", "images_unannotated_src"]
        for i in range(1, 101)
    }
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and distributed(path))
    actual_images = {path.relative_to(ROOT).as_posix() for path in files
                     if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}}
    assert actual_images == expected_images, "Unexpected or missing distributed raster assets"
    assert not any(path.suffix.lower() in {".pth", ".pt", ".safetensors", ".docx", ".tex"} for path in files)
    assert not any(path.is_symlink() for path in files), "Release files must be self-contained"
    manifest = list(csv.DictReader((ROOT / "panel_manifest.csv").open()))
    assert len(manifest) == 15
    for row in manifest:
        assert (ROOT / row["script"]).is_file()
        for relative in row["data_inputs"].split(";"):
            assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
            assert (ROOT / relative).exists(), f"Missing panel input {relative}"
    python_count = 0
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=path.relative_to(ROOT).as_posix())
            python_count += 1
        if path.suffix in {".py", ".md", ".txt", ".json", ".csv", ".cff"}:
            content = path.read_text(encoding="utf-8")
            assert not re.search(r"/(?:Users|home)/[A-Za-z0-9_.-]+/", content), "Personal absolute path: " + str(path.relative_to(ROOT))
            assert not re.search(r"(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,})", content), "Potential credential"
    checksum_entries = 0
    if not args.skip_checksums:
        checksum_file = ROOT / "SHA256SUMS.txt"
        for line in checksum_file.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
            assert sha(ROOT / relative) == digest, "SHA256 mismatch: " + relative
            checksum_entries += 1
        expected = {path.relative_to(ROOT).as_posix() for path in files if path != checksum_file}
        present = {line.split("  ", 1)[1] for line in checksum_file.read_text().splitlines()}
        assert present == expected, "Checksum manifest does not cover exactly the distributed files"
    report = {"status": "PASS", "samples": len(records), "identities": len({r["identity_id"] for r in records}),
              "unannotated_rasters": len(actual_images), "json_annotation_files": len(records),
              "annotations": len(points), "type_counts": dict(counts), "panel_entries": len(manifest),
              "python_sources_parsed": python_count, "checksum_entries": checksum_entries,
              "unexpected_raster_assets": [], "personal_paths_or_credentials_found": False,
              "annotation_overlay_created": False, "scope": "File integrity, cohort, coordinates and input contracts"}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
