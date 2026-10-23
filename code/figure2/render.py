#!/usr/bin/env python3
"""Draw Figure 2 summary bars and paired reconstruction galleries.
Use --private-assets-dir for externally supplied illustration files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
INK, EDGE = "#263944", "#2D4B59"
BLUE = ["#A4B4CC", "#6B83A8", "#3C5488"]
RED = ["#F0B2B2", "#E07070", "#DC0000"]
PALETTE = {"source": "#F3F0EA", "torso": "#EAF1F6", "records": "#EAF3EF"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style():
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .65, "xtick.major.width": .65,
        "ytick.major.width": .65, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.direction": "out",
        "ytick.direction": "out", "pdf.fonttype": 42,
        "svg.fonttype": "none", "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": EDGE,
        "savefig.facecolor": "white"})


def ax_mm(fig, box, size):
    x, y, w, h = box
    W, H = size
    return fig.add_axes([x/W, y/H, w/W, h/H])


def text_mm(fig, x, y, text, size, **kwargs):
    return fig.text(x/size[0], y/size[1], text, **kwargs)


def letter(ax, value):
    ax.text(.025, .98, value, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", zorder=40,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": .2})


def quantitative_a(fig, values, box, size):
    ax = ax_mm(fig, box, size)
    x = np.arange(3)
    bars = ax.bar(x, values["means_A"], yerr=values["stds_A"], width=.55,
        color=BLUE, edgecolor=INK, linewidth=.65, capsize=2.5,
        error_kw={"elinewidth": .7, "capthick": .7})
    for i, (mean, error) in enumerate(zip(values["means_A"], values["stds_A"])):
        ax.text(i, mean+error+1.8, f"{mean:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, ["GrabCut", "Mask\nR-CNN", "OpenStripe\n(Ours)"])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylabel("mIoU (%)", labelpad=2)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(0, 110)
    ax.set_xlim(-.65, 2.65)
    ax.grid(axis="y", color="#E5EBEE", lw=.45)
    ax.set_axisbelow(True)
    letter(ax, "a")
    np.testing.assert_array_equal([b.get_height() for b in bars], values["means_A"])
    segments = ax.containers[0].lines[2][0].get_segments()
    np.testing.assert_allclose([[p[0,1], p[1,1]] for p in segments],
        np.c_[np.array(values["means_A"])-values["stds_A"],
                 np.array(values["means_A"])+values["stds_A"]], rtol=0, atol=1e-13)
    return ax


def quantitative_b(fig, values, box, size):
    ax = ax_mm(fig, box, size)
    right = ax.twinx()
    x, width = np.arange(3), .31
    for target, offset, key, errors, colors, delta in [
        (ax, -width/2, "pck_means", "pck_stds", BLUE, 1.8),
        (right, width/2, "nme_means", "nme_stds", RED, .17)]:
        bars = target.bar(x+offset, values[key], yerr=values[errors],
            width=width, color=colors, edgecolor=INK, linewidth=.65,
            capsize=2, error_kw={"elinewidth": .7, "capthick": .7})
        for i, bar in enumerate(bars):
            target.text(bar.get_x()+bar.get_width()/2,
                values[key][i]+values[errors][i]+delta, f"{values[key][i]:.1f}",
                ha="right" if target is ax else "left", va="bottom",
                fontsize=8.5, color=colors[-1])
        np.testing.assert_array_equal([b.get_height() for b in bars], values[key])
        segments = target.containers[0].lines[2][0].get_segments()
        np.testing.assert_allclose([[p[0,1], p[1,1]] for p in segments],
            np.c_[np.array(values[key])-values[errors],
                     np.array(values[key])+values[errors]], rtol=0, atol=1e-13)
    ax.set_xticks(x, ["TPS", "STN", "OpenStripe\n(Ours)"])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylabel("PCK@0.1 (%) ↑ Higher is better", color=BLUE[-1], labelpad=2)
    right.set_ylabel("NME (%) ↓ Lower is better", color=RED[-1], labelpad=2)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(0, 110)
    ax.set_xlim(-.65, 2.65)
    right.set_yticks(np.arange(0, 11, 2))
    right.set_ylim(0, 11)
    ax.tick_params(axis="y", colors=BLUE[-1])
    right.tick_params(axis="y", colors=RED[-1])
    ax.spines["left"].set_color(BLUE[-1])
    for name, spine in right.spines.items():
        spine.set_visible(name == "right")
    right.spines["right"].set_color(RED[-1])
    ax.grid(axis="y", color="#E5EBEE", lw=.45)
    ax.set_axisbelow(True)
    letter(ax, "b")
    return ax, right


def frame(fig, box, size, face="none", lw=1.1, zorder=20):
    x, y, w, h = box
    W, H = size
    patch = Rectangle((x/W, y/H), w/W, h/H, transform=fig.transFigure,
        facecolor=face, edgecolor=EDGE, lw=lw, zorder=zorder)
    fig.add_artist(patch)


def source_image(path, trim=False):
    with Image.open(path) as im:
        im = im.convert("RGB")
    if trim:
        a = np.asarray(im)
        ink = a.min(axis=2) < 160
        ink[:3] = ink[-3:] = False
        ink[:, :3] = ink[:, -3:] = False
        yy, xx = np.nonzero(ink)
        if len(xx):
            margin = max(3, round(max(im.size)*.008))
            im = im.crop((max(0, int(xx.min())-margin),
                max(0, int(yy.min())-margin), min(im.width, int(xx.max())+margin+1),
                min(im.height, int(yy.max())+margin+1)))
    # Downsample the rendered derivative only; source data remain untouched.
    im.thumbnail((1600, 1000), Image.Resampling.LANCZOS)
    return im


def image_tile(fig, path, box, size, header, color, trim=False, counts=None):
    x, y, w, h = box
    head, foot = 3.5, 3.8 if counts is not None else 0
    frame(fig, box, size, face="white", zorder=-2)
    frame(fig, (x, y+h-head, w, head), size, face=color, lw=.5, zorder=1)
    text_mm(fig, x+.7, y+h-head/2, header, size, fontsize=7.1,
            ha="left", va="center", zorder=30)
    im = source_image(path, trim)
    scale = min((w-.8)/im.width, (h-head-foot-.8)/im.height)
    iw, ih = im.width*scale, im.height*scale
    image_ax = ax_mm(fig, (x+(w-iw)/2, y+foot+(h-head-foot-ih)/2, iw, ih), size)
    image_ax.imshow(im, interpolation="none")
    image_ax.axis("off")
    if counts is not None:
        text_mm(fig, x+w/2, y+foot/2, counts, size, fontsize=7.0,
                ha="center", va="center", zorder=30)
    frame(fig, box, size)
    return dict(file=path.name, source_sha256=sha(path), frame_mm=list(box),
                rendered_trim=trim, annotation_overlay=False)


def gallery(fig, root, size, x_offset=0, private_assets_dir=None):
    """Six paired rows and four right-column cards, all native artists."""
    samples = {r["sample_id"]: r for r in csv.DictReader(
        (root / "data/samples.csv").open(encoding="utf-8"))}
    config = json.loads((root / "data/figure2/public_gallery_samples.json").read_text())
    width1, width2, width3, gap = 49.4, 49.4, 66.8, .9
    x1, x2, x3 = 69.1-x_offset, 119.4-x_offset, 169.7-x_offset
    top, bottom, header, gapy = 165.5, 3.1, 5.2, .8
    gallery_top = top-header-1.0
    private = private_assets_dir is not None
    headers = (["Camera images", "Stripe reconstructions", "Structural features"]
               if private else ["Unmarked full-body patterns", "Unmarked stripe patterns", "Structural records (JSON)"])
    colors = [PALETTE["source"], PALETTE["torso"], PALETTE["records"]]
    for x, width, label, color in zip([x1,x2,x3], [width1,width2,width3], headers, colors):
        frame(fig, (x, top-header, width, header), size, face=color, lw=.8, zorder=1)
        text_mm(fig, x+width/2, top-header/2, label, size, fontsize=7.3,
                fontweight="bold", ha="center", va="center", zorder=30)
    text_mm(fig, x1+.5, top+1.3, "c", size, fontsize=10,
            fontweight="bold", ha="left", va="center", zorder=40)
    layout = []
    h = (gallery_top-bottom-5*gapy)/6
    for i, sid in enumerate(config["paired_rows"]):
        y = gallery_top-(i+1)*h-i*gapy
        r = samples[sid]
        if private:
            alias = ["A1","A2","A3","B","C","D"][i]
            p1, p2 = private_assets_dir/f"{alias}_camera.png", private_assets_dir/f"{alias}_reconstruction.png"
            labels = [alias+" | Camera", alias+" | Reconstruction"]
        else:
            p1, p2 = root/r["src_image"], root/r["common_bw_image"]
            labels = [f'{sid} | {r["identity_id"]} | Full body', f'{sid} | {r["identity_id"]} | Common BW']
        for x, w, path, label, color in zip([x1,x2], [width1,width2], [p1,p2], labels, colors[:2]):
            layout.append(image_tile(fig, path, (x,y,w,h), size, label, color, trim=not private))
    h = (gallery_top-bottom-3*gapy)/4
    private_counts = json.loads((root/"data/figure2/private_illustration_counts.json").read_text())
    for i, sid in enumerate(config["structural_records"]):
        y = gallery_top-(i+1)*h-i*gapy
        if private:
            alias = "ABCD"[i]
            p = private_assets_dir/f"{alias}_topology.png"
            label = f'{alias}+ | E {private_counts[i]["endpoints"]}; F {private_counts[i]["bifurcations"]}'
            count_text = None
        else:
            r = samples[sid]
            p = root/r["common_bw_image"]
            annotation = json.loads((root/r["annotation"]).read_text())
            counts = {t: sum(a["type"] == t for a in annotation["annotations"]) for t in "BFIC"}
            assert all(counts[t] == int(r[f"primary_{t}_count"]) for t in "BFIC")
            label = f'{sid} | {r["identity_id"]}'
            count_text = "   ".join(f"{t} = {counts[t]}" for t in "BFIC")
        layout.append(image_tile(fig, p, (x3,y,width3,h), size, label,
                                 colors[2], trim=not private, counts=count_text))
    assert len(layout) == 16
    for tile in layout:
        tile["annotation_overlay"] = bool(private)
    return layout


def render(panel, output_dir, root=ROOT, private_assets_dir=None, dpi=300, formats=("pdf", "png", "svg")):
    style()
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = json.loads((root/"data/figure2/archived_plot_values.json").read_text())
    if private_assets_dir is not None:
        private_assets_dir = Path(private_assets_dir)
        expected = [f"{a}_{k}.png" for a in ["A1","A2","A3","B","C","D"] for k in ["camera","reconstruction"]]
        expected += [f"{a}_topology.png" for a in "ABCD"]
        missing = [n for n in expected if not (private_assets_dir/n).is_file()]
        if missing:
            raise FileNotFoundError("Missing private prepared assets: "+", ".join(missing))
    size = {"a":(74.,88.), "b":(88.,88.), "c":(174.,168.5), "all":(240.,168.5)}[panel]
    fig = plt.figure(figsize=(size[0]/25.4,size[1]/25.4),dpi=200,facecolor="white")
    layout = []
    if panel in ["a","all"]:
        quantitative_a(fig,values,(11.,93.6,52.,71.8) if panel=="all" else (13.,12.,58.,72.),size)
    if panel in ["b","all"]:
        quantitative_b(fig,values,(11.,10.,44.2,71.3) if panel=="all" else (16.,12.,55.,72.),size)
    if panel in ["c","all"]:
        layout = gallery(fig,root,size,x_offset=66.8 if panel=="c" else 0,
                         private_assets_dir=private_assets_dir)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside=[]
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.width>.1 and box.height>.1 and (box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5):
                outside.append(text.get_text())
    if outside:
        raise ValueError("Labels extend outside figure: "+repr(outside))
    suffix = "" if panel=="all" else panel
    private_suffix = "_private" if private_assets_dir is not None else ""
    name = "Figure2"+suffix+private_suffix
    outputs=[]
    for ext in formats:
        path=output_dir/f"{name}.{ext}"
        kwargs={"dpi":dpi} if ext=="png" else {}
        if ext=="pdf":kwargs["metadata"]={"Title":name+"; native redraw", "CreationDate":None,"ModDate":None}
        fig.savefig(path,**kwargs)
        outputs.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)))
    plt.close(fig)
    meta=dict(panel=panel,size_mm=list(size),renderer="native Matplotlib artists; no PDF crop",
        mode="private original illustrations" if private_assets_dir is not None else "public unmarked reconstructions and numeric JSON records",
        quantitative_source="archived summary means and error-bar lengths; individual evaluation records and error definition unavailable",
        inferential_tests_recomputed=False,significance_values_invented=False,
        source_values_sha256=sha(root/"data/figure2/archived_plot_values.json"),
        source_values=values,unmarked_image_layout=layout,outside_labels=outside,outputs=outputs)
    (output_dir/f"{name}_render.json").write_text(json.dumps(meta,indent=2)+"\n")
    return meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel",choices=["a","b","c","all"],default="all")
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--output-dir",type=Path,default=ROOT/"build/figure2")
    ap.add_argument("--private-assets-dir",type=Path)
    ap.add_argument("--dpi",type=int,default=300)
    ap.add_argument("--formats",default="pdf,png,svg")
    args=ap.parse_args()
    formats=[s.strip() for s in args.formats.split(",") if s.strip()]
    if not formats or any(s not in {"pdf","png","svg"} for s in formats):
        ap.error("--formats must be a comma-separated selection of pdf,png,svg")
    panels=["a","b","c","all"] if args.panel=="all" else [args.panel]
    results=[render(p,args.output_dir,args.root,args.private_assets_dir,args.dpi,formats) for p in panels]
    print(json.dumps(dict(status="PASS",panels=[r["panel"] for r in results],
        outputs=[o["file"] for r in results for o in r["outputs"]],
        private_assets_used=args.private_assets_dir is not None)))


if __name__=="__main__":
    main()
