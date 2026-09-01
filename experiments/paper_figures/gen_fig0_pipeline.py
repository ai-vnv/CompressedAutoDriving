"""Fig. 0 — full pipeline overview with real simulator footage.

Story, left to right: POMDP simulator -> onboard observation -> perception (YOLO
objects, MobileNetV3-small lane pose) -> recursive belief b_t in R^29 -> Belief-PPO
curriculum training -> actor extraction -> compression story (prune collapse, KD
coverage split, precision fan-out) -> evaluation harness under every stage.

All numbers shown are verified against repository artifacts (integrity_phase_c.sh).
Camera insets are cropped from same-seed primary-rollout contact sheets in
paper/figure_sources/ (camera area only, overlay removed; provenance in the caption).
No title inside the figure; the caption lives in LaTeX.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from paper_plot_style import BLUE, DARK, GREEN, VERMILLION, save_fig

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper" / "figure_sources"

W, H = 200.0, 104.0
fig, ax = plt.subplots(figsize=(7.16, 3.2))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
UX, UY = 200 / 7.16, 104 / 3.2  # units per inch on each axis


def box(x0, y0, x1, y1, text, fc="white", ec=DARK, fs=5.2, weight="normal", tc="black", lw=0.8):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.15,rounding_size=0.9",
                                fc=fc, ec=ec, lw=lw))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, linespacing=1.3)


def tag(x0, y0, x1, y1, text, color=VERMILLION, fs=4.9):
    box(x0, y0, x1, y1, text, fc=color, ec=color, fs=fs, weight="bold", tc="white")


def arrow(p0, p1, lw=0.8, color=DARK):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6,
                                 lw=lw, color=color))


def note(x, y, text, fs=4.4, color="#555555"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            style="italic", linespacing=1.3)


def camera_crop(sheet, row, col):
    """Camera-only area of one contact-sheet tile (overlay bar removed)."""
    img = Image.open(sheet)
    tw, th = img.width // 4, img.height // 2
    skip = int(th * 0.26)
    return img.crop((col * tw, row * th + skip, (col + 1) * tw, (row + 1) * th))


def inset(x0, y0, w_units, image, label=None):
    h_units = w_units * (image.height / image.width) * (UY / UX)
    x1, y1 = x0 + w_units, y0 + h_units
    ax.imshow(image, extent=(x0, x1, y0, y1), aspect="auto", zorder=3)
    ax.add_patch(Rectangle((x0, y0), w_units, h_units, fill=False, ec=DARK, lw=0.7, zorder=4))
    if label:
        note((x0 + x1) / 2, y0 - 3.2, label, fs=4.2)
    return x1, y1


# ------------------------------------------------------------------ perception column
box(2, 92, 46, 102, "Gym-Duckietown simulator --- POMDP $(\mathcal{S},\mathcal{A},T,R,\Omega,O,\gamma)$", fs=5.2)
world = camera_crop(SRC / "figT_c2_A0_seed180207_contactsheet.png", 0, 0)
_, wtop = inset(14, 74, 20, world)
note(24, 71.2, "$o_t$: onboard camera")
arrow((24, 92), (24, wtop + 0.5))

box(2, 58, 22, 68, "YOLO\npedestrians, stop signs", fs=4.6)
box(26, 58, 46, 68, "MobileNetV3-small\nlane pose", fs=4.6)
arrow((20, 74), (13, 68.4)); arrow((28, 74), (35, 68.4))

box(6, 38, 42, 54, "recursive belief update\n$b_t \\in \\mathbb{R}^{29}$: existence probs.,\nmeans, dispersions, stop mode", fs=4.9)
arrow((12, 58), (18, 54.4)); arrow((35, 58), (30, 54.4))

# ------------------------------------------------------------------ training column
box(48, 90, 80, 100, "Belief-PPO training", fs=5.4)
cw = 32 / 5.0
for i, c in enumerate(["C0", "C1", "C2", "C3", "C4"]):
    box(48 + i * cw + 0.4, 81, 48 + (i + 1) * cw - 0.4, 87.5, c, fc="#ECEFF1", fs=4.8)
arrow((64, 90), (64, 88))

box(48, 50, 80, 70, "actor $\\mathbf{A0}$\n29--256--256--2 tanh\n73,986 params, FP32", fc="#E3F2FD", ec=BLUE, fs=5.2)
arrow((42, 48), (48, 56))
ax.text(44.2, 54.2, "$b_t$", fontsize=5.2, ha="center")
arrow((64, 81), (64, 70.4))

# ------------------------------------------------------------------ prune
box(88, 50, 116, 70, "structured pruning\nwidth 64\n6,210 params, $-$91.6%", fs=5.2)
arrow((80, 60), (88, 60))
tag(86, 42.5, 118, 48.5, "A1: loses all 5 curricula", fs=4.6)

# ------------------------------------------------------------------ KD split
box(124, 80, 158, 90, "KD, C4-focused rehearsal", fs=4.9)
tag(124, 72.5, 158, 78.5, "A2: restores C3--C4 only", fs=4.6)
box(124, 51, 158, 63, "KD, balanced C0--C4\nrehearsal, 62,176 states", fs=4.9)
tag(124, 43.5, 158, 49.5, "A3: restores all 5", GREEN)
arrow((116, 65), (124, 84)); arrow((116, 57), (124, 57))

arrow((158, 57), (168, 57), lw=1.0)
note(162, 48.5, "same fixed checkpoint,\nno training below", fs=4.2)

# ------------------------------------------------------------------ precision fan-out
box(172, 86, 198, 93.5, "PTQ INT8 (A6)", fs=5.2)
tag(172, 79, 198, 85, "C3, C4 fail:\nfreeze at stop line", VERMILLION, fs=4.5)
box(172, 60, 198, 67.5, "QAT+KD INT8 (A8)", fs=5.2)
tag(172, 53, 198, 59, "C3, C4 fail:\ndrives through stop", VERMILLION, fs=4.5)
note(185, 50, "no footage; telemetry only", fs=4.0, color="#888888")
box(172, 34, 198, 41.5, "FP16 cast (A9)", fs=5.2)
tag(172, 27, 198, 33, "all 5 curricula\nretained", GREEN, fs=4.5)

for y in (89.5, 63.5, 37.5):
    arrow((168, 57), (172, y))

note(146, 100, "control: PTQ of the unpruned width-256 actor (A4) passes all 5", fs=4.4)

# ------------------------------------------------------------------ evidence strip
crash = camera_crop(SRC / "fig0_A1_c2_seed180201_lanefailure_contactsheet.png", 1, 3)
_, ctop = inset(90, 24, 22, crash)
note(101, 20.6, "A1 leaves the lane")
freeze = camera_crop(SRC / "figT_c3_A6_seed180201_contactsheet.png", 1, 3)
inset(124, 24, 22, freeze)
note(135, 20.6, "A6 parked at the line, $v{=}0$, timeout")
note(123, 13.5, "frames from the same-seed primary rollouts", fs=4.2)

# ------------------------------------------------------------------ evaluation harness
box(88, 2, 198, 10,
    "evaluation after every stage: same 8 seeds $\\times$ 5 curricula,"
    " checks fixed in advance, bit-for-bit reproducible backend",
    fc="#ECEFF1", ec="#90A4AE", fs=4.9)
for x in (102, 141, 185):
    arrow((x, 10), (x, 13.5), lw=0.7, color="#78909C")

save_fig(fig, "fig0_pipeline")
