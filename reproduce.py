#!/usr/bin/env python3
"""Draw nonspatial main-text panels from the released inputs."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PANELS = {
    "1": ["workflow"],
    "2": list("abc"),
    "3": ["a", "b", "c1", "c2", "c3", "c4", "d1", "d2", "d3", "d4", "e"],
}


def draw(figure, panel, output, formats, dpi):
    module = ROOT / "code" / ("figure" + figure)
    script = module / "render.py"
    cmd = [sys.executable, str(script), "--panel", panel,
           "--output-dir", str(output),
           "--dpi", str(dpi), "--formats"]
    cmd += formats if figure == "3" else [",".join(formats)]
    print(f"Drawing Figure {figure}, panel {panel}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="List all 15 nonspatial panel entries")
    mode.add_argument("--all", action="store_true", help="Draw all released panels and combined figures")
    mode.add_argument("--figure", choices=PANELS)
    p.add_argument("--panel", default="all", help="Panel label, or all for panels plus complete figure")
    p.add_argument("--output-dir", type=Path, default=ROOT / "generated")
    p.add_argument("--formats", default="pdf,png,svg", help="Comma-separated pdf,png,svg")
    p.add_argument("--dpi", type=int, default=240)
    args = p.parse_args()
    if args.list:
        print(json.dumps(PANELS, indent=2))
        return
    formats = list(dict.fromkeys(s.strip() for s in args.formats.split(",") if s.strip()))
    if not formats or any(s not in {"pdf", "png", "svg"} for s in formats):
        p.error("--formats must contain pdf, png and/or svg")
    if args.dpi < 20:
        p.error("--dpi must be at least 20")
    output = args.output_dir.resolve()
    figures = list(PANELS) if args.all else [args.figure]
    for figure in figures:
        panel = "all" if args.all else args.panel
        permitted = [*PANELS[figure], "all"]
        if figure == "3":
            permitted += ["c", "d"]
        if panel not in permitted:
            p.error(f"Figure {figure} panel must be one of {permitted}")
        folder = output / ("figure" + figure)
        if panel == "all":
            # Figure 2's own all command also creates its three separate panels.
            jobs = ["all"] if figure in {"1", "2"} else [*PANELS[figure], "all"]
        else:
            jobs = [panel]
        for job in jobs:
            draw(figure, job, folder, formats, args.dpi)
    print(json.dumps({"status": "PASS", "figures": figures, "output_dir": str(output),
                      "original_camera_photos_used": False, "annotated_rasters_used": False}))


if __name__ == "__main__":
    main()
