"""Draw the Figure 1 workflow from unannotated images, JSON cards and schematic clustering dots."""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault('MPLCONFIGDIR', str(Path.home() / '.cache/openstripe/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W = H = 180.0
INK, MUTED = '#24394A', '#5B7180'
BLUE, TEAL, ORANGE, PURPLE = '#0072B2', '#009E73', '#D55E00', '#8261A8'
COLORS = dict(zip('BFIC', ['#E41A1C', '#C29920', '#009CCB', '#588F31']))


def box(ax, x, y, w, h, fill='white', edge='#CADAE3', lw=.7, radius=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=fill, edgecolor=edge, linewidth=lw))


def text(ax, x, y, value, size=8.2, color=INK, bold=False, ha='center', **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color,
        fontweight='bold' if bold else 'normal', ha=ha, va='center', **kwargs)


def arrow(ax, start, end, color=BLUE, lw=.9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
        mutation_scale=8, color=color, linewidth=lw, shrinkA=0, shrinkB=0))


def stripe_image(ax, path, x, y, w, h):
    with Image.open(path) as source:
        a = np.asarray(source.convert('RGB'))
    # Trim the surrounding near-white canvas for display.
    yy, xx = np.where(np.min(a, axis=2) < 200)
    if len(xx):
        pad = 8
        a = a[max(0, yy.min()-pad):min(a.shape[0], yy.max()+pad+1),
              max(0, xx.min()-pad):min(a.shape[1], xx.max()+pad+1)]
    scale = min(w/a.shape[1], h/a.shape[0])
    iw, ih = a.shape[1]*scale, a.shape[0]*scale
    ax.imshow(a, extent=(x+(w-iw)/2, x+(w+iw)/2,
        y+(h-ih)/2, y+(h+ih)/2), origin='upper', interpolation='none', zorder=2)


def annotation_card(ax, record, x, y, w, h):
    box(ax, x, y, w, h, '#FFFCF7', '#DFBA9C')
    sid = record['sample_id']
    text(ax, x+3, y+h-4, sid+'.json', 8.4, ORANGE, True, 'left')
    text(ax, x+3, y+h-8.3, 'Machine-readable coordinates', 7.0, MUTED, ha='left')
    points = record['annotations']
    for i, kind in enumerate('BFIC'):
        selected = [p for p in points if p['type'] == kind]
        yy = y+h-13.5-i*4.9
        text(ax, x+4, yy, kind, 8, COLORS[kind], True, 'left')
        if selected:
            coords = ', '.join(f"({p['x_normalized']:.3f}, {p['y_normalized']:.3f})"
                               for p in selected[:2])
            if len(selected) > 2:
                coords += ', ...'
        else:
            coords = '[]'
        text(ax, x+10, yy, coords, 6.5, INK, ha='left', family='monospace')


def draw(config, output, formats, private_assets=None, dpi=240):
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'], 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'font.size': 8})
    fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H), aspect='equal')
    ax.axis('off')
    box(ax, 2, 34, 176, 144, '#F3F8FB', '#78ABC4', 1.0, 2)
    box(ax, 6, 165, 38, 10, '#EDF4F9', '#AFCADA')
    ax.add_patch(Rectangle((8, 168), 7, 4, facecolor='#DCE9F2', edgecolor=BLUE, lw=.7))
    ax.add_patch(Circle((11.5, 170), 1.4, facecolor='white', edgecolor=BLUE, lw=.7))
    text(ax, 17, 171, 'Camera images', 8, BLUE, True, 'left')
    text(ax, 17, 167.5, 'Schematic input', 6.3, MUTED, ha='left')
    arrow(ax, (45, 170), (50, 170))
    box(ax, 51, 165, 123, 10, '#E5F2FA', '#5D9FC4')
    text(ax, 56, 171, 'OpenStripe', 12, BLUE, True, 'left')
    text(ax, 100, 171, 'four-agent system', 9, BLUE, True, 'left')
    text(ax, 56, 167.3, 'Canonical stripe reconstruction and structural annotation', 7, MUTED, ha='left')
    ax.plot([25, 154], [162.8, 162.8], lw=.7, color=BLUE)
    ax.plot([112.5, 112.5], [165, 162.8], lw=.7, color=BLUE)
    shades = ['#E8F3FA', '#E8F5EE', '#FFF0E5', '#F1EBF7']
    for i, role in enumerate(config['roles']):
        x = 6+43*i
        color = [BLUE, TEAL, ORANGE, PURPLE][i]
        arrow(ax, (x+19.5, 162.8), (x+19.5, 161), color)
        box(ax, x, 142, 39, 19, shades[i], color)
        text(ax, x+19.5, 156.7, role['title'], 7.6, color, True)
        text(ax, x+19.5, 150.7, role['description'], 6.7, MUTED)
        text(ax, x+19.5, 146.2, role['method'], 7.1, color, True)
        arrow(ax, (x+17.5, 142), (x+17.5, 138), color, .6)
        arrow(ax, (x+21.5, 138), (x+21.5, 142), color, .6)
    box(ax, 6, 133, 168, 5, '#DDECF4', '#8BB1C6')
    text(ax, 90, 135.5, 'Shared workspace (blackboard)', 8, '#305F77', True)
    text(ax, 42, 127.5, 'Unannotated stripe reconstructions', 8.4, TEAL, True)
    text(ax, 138, 127.5, 'Structural annotation records', 8.4, ORANGE, True)
    for sid, y, height in [('S016', 84, 39), ('S066', 42, 39)]:
        rec = json.loads((ROOT/'data/annotations'/f'{sid}.json').read_text())
        box(ax, 6, y, 72, height, 'white', '#BAD7CC')
        stripe_image(ax, ROOT/rec['image']['file'], 8, y+5, 68, height-7)
        text(ax, 9, y+2.8, f"{sid} / {rec['identity_id']}", 7.2, bold=True, ha='left')
        if private_assets is None:
            annotation_card(ax, rec, 102, y, 72, height)
        else:
            path = private_assets/f'{sid}_annotated.png'
            if not path.is_file():
                raise FileNotFoundError('Explicit private asset required: '+str(path))
            box(ax, 102, y, 72, height, 'white', '#DFBA9C')
            stripe_image(ax, path, 104, y+5, 68, height-7)
            text(ax, 105, y+2.8, sid+' / private paper asset', 6.9, ORANGE, ha='left')
        text(ax, 90, y+height/2+4, 'OpenStripe', 7.3, ORANGE, True)
        text(ax, 90, y+height/2, 'adds records', 6.6, ORANGE, True)
        arrow(ax, (80, y+height/2-5), (100, y+height/2-5), ORANGE)
    box(ax, 2, 2, 176, 30, '#EFF3F8', '#A3B7CB', .9)
    arrow(ax, (90, 41), (90, 33), TEAL)
    for x, width, title in [(6, 45, 'Feature vectors'), (63, 55, 'Clustering algorithm'), (130, 44, 'Identity groups')]:
        box(ax, x, 6, width, 20, 'white', '#83A8BC')
        text(ax, x+width/2, 22, title, 8.4, BLUE, True)
    for j, kind in enumerate('BFIC'):
        text(ax, 14+j*9.5, 13, kind, 10, COLORS[kind], True)
    rng = np.random.default_rng(13)
    for j, color in enumerate([BLUE, ORANGE, TEAL]):
        xy = rng.normal(size=(5, 2))*.75+[74+j*16, 13]
        ax.scatter(xy[:, 0], xy[:, 1], s=9, color=color)
        ax.add_patch(Circle((138+j*14, 13), 3, facecolor=color, edgecolor='white'))
        text(ax, 138+j*14, 13, str(j+1), 7, 'white', True)
    text(ax, 90, 8, 'schematic', 5.8, MUTED)
    arrow(ax, (52, 15), (62, 15))
    arrow(ax, (119, 15), (129, 15))
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        target = output/f'Figure1_workflow.{ext}'
        fig.savefig(target, dpi=dpi, metadata={'Title': 'OpenStripe public workflow'} if ext == 'pdf' else None)
        written.append(str(target))
    plt.close(fig)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', choices=['all', 'workflow'], default='all')
    p.add_argument('--output-dir', type=Path, default=ROOT/'build/figure1')
    p.add_argument('--formats', default='pdf,png,svg')
    p.add_argument('--dpi', type=int, default=240)
    p.add_argument('--private-assets-dir', type=Path,
        help='Optional directory of externally supplied illustration images.')
    args = p.parse_args()
    config = json.loads((ROOT/'data/figure1/workflow.json').read_text())
    files = draw(config, args.output_dir, args.formats.split(','), args.private_assets_dir, args.dpi)
    print(json.dumps({'figure': 1, 'public_image_mode': args.private_assets_dir is None, 'files': files}))


if __name__ == '__main__':
    main()
