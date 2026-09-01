"""fig:task — one scene render per curriculum (capture_task_views.py).

C0 and C1 are bird's-eye views of the full training maps with the robot on the
lane. C2-C4 are elevated third-person renders of the evaluation scenes: the
ego robot together with the crossing duckie mid-lane (C2), the stop sign at
the stop location (C3), and both hazards along the same lap (C4).
Double-column figure.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from paper_plot_style import save_fig

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper" / "figure_sources" / "scene_views"

# (label, file, crop box left/top/right/bottom)
TILES = [
    ("C0  lane following", "c0_smallloop_bev.png", (30, 30, 870, 870)),
    ("C1  larger loop", "c1_experimentloop_bev.png", (30, 30, 870, 870)),
    ("C2  crossing pedestrian", "c2_crossing.png", (60, 300, 900, 930)),
    ("C3  stop compliance", "c3_stop.png", (240, 330, 1240, 1080)),
    ("C4  combined", "c4_combined.png", (120, 390, 1400, 890)),
]

ratios = []
for _, name, box in TILES:
    ratios.append((box[2] - box[0]) / (box[3] - box[1]))

fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.30),
                         gridspec_kw={"width_ratios": ratios})
for ax, (label, name, box) in zip(axes, TILES):
    ax.imshow(Image.open(SRC / name).crop(box))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.set_xlabel(label, fontsize=6.6, labelpad=2.5)
fig.subplots_adjust(wspace=0.05, left=0.003, right=0.997, top=0.99, bottom=0.17)
save_fig(fig, "fig_task")
