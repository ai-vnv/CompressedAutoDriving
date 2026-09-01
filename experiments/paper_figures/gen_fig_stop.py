"""fig:phenotypes — A6 freeze scene reconstruction + stop-approach telemetry.

Top: four third-person scene renders of the A6 C3 episode (seed 180201). The
ego pose in each is reconstructed from the recorded per-step telemetry
(stop_line_distance_m along the northbound approach); the camera is fixed so
the approach and the freeze are directly comparable. Labels give the step and
the commanded velocity at that step. Frames and their (step, v, d) records are
produced by capture_task_views.py freeze into figure_sources/scene_views/.
Bottom: commanded velocity and stop-line distance versus step, read from the
same telemetry. A3 and A6 use seed 180201; A8's lowest failing seed is 180203
and is plotted with that seed (stated in caption).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from paper_plot_style import BLUE, VERMILLION, save_fig

ROOT = Path(__file__).resolve().parents[2]
TEL = ROOT / "artifacts/f17_optimization_method_order_v1/telemetry"
SCENES = ROOT / "paper" / "figure_sources" / "scene_views"

STOP_IDX = 9  # stop_line_distance_m in public_physical_29d (verified order)


def trace(pid, seed):
    with np.load(TEL / pid / "c3" / f"seed_{seed}" / "trace.npz",
                 allow_pickle=False) as z:
        names = [str(v) for v in z["feature_names"]]
        assert names[STOP_IDX] == "stop_line_distance_m"
        return (np.asarray(z["physical_action"], dtype=np.float32)[:, 0],
                np.asarray(z["public_physical_29d"], dtype=np.float32)[:, STOP_IDX])


series = {
    "A3 (FP32 parent)": (*trace("A3", 180201), BLUE, "-"),
    "A6 (PTQ)": (*trace("A6", 180201), VERMILLION, "-"),
    "A8 (QAT, seed 180203)": (*trace("A8", 180203), VERMILLION, "--"),
}
frames = json.loads((SCENES / "a6_freeze_frames.json").read_text())

fig = plt.figure(figsize=(3.5, 4.05))
gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 2.0], hspace=0.26)
gs_top = gs[0].subgridspec(1, 4, wspace=0.06)

# ---- scene reconstruction strip ---------------------------------------------
CROP = (380, 330, 1180, 1080)
for k, rec in enumerate(frames):
    ax = fig.add_subplot(gs_top[k])
    ax.imshow(Image.open(SCENES / rec["file"]).crop(CROP))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.set_xlabel(f"step {rec['step']}\n$v$ = {rec['v_cmd_mps']:.3f} m/s",
                  fontsize=6.0, labelpad=1.8)

gs_bot = gs[1].subgridspec(2, 1, hspace=0.14)

# ---- telemetry panels --------------------------------------------------------
ax1 = fig.add_subplot(gs_bot[0])
ax2 = fig.add_subplot(gs_bot[1], sharex=ax1)
for label, (v, d, color, ls) in series.items():
    steps = np.arange(len(v))
    ax1.plot(steps, v, color=color, ls=ls, lw=1.0, label=label)
    ax2.plot(steps, d, color=color, ls=ls, lw=1.0)
# mark the reconstructed steps on the A6 curves
va6, da6, _, _ = series["A6 (PTQ)"]
mark = [rec["step"] for rec in frames]
ax1.plot(mark, va6[mark], "o", color=VERMILLION, ms=3.4, mfc="none", mew=0.9)
ax2.plot(mark, da6[mark], "o", color=VERMILLION, ms=3.4, mfc="none", mew=0.9)
ax1.set_ylabel("$v_{\mathrm{cmd}}$ (m/s)", fontsize=7)
ax1.tick_params(labelsize=6.5)
fig.legend(fontsize=5.6, frameon=False, ncol=3, loc="center",
           bbox_to_anchor=(0.57, 0.660), columnspacing=0.9, handlelength=1.4)
plt.setp(ax1.get_xticklabels(), visible=False)
ax2.axhline(0.0, color="#555555", lw=0.7, ls=":")
ax2.text(60, -0.42, "stop line", fontsize=5.8, color="#555555", ha="left")
ax2.set_ylim(-1.7, 0.8)
ax2.annotate("A3 drives on (completes)", xy=(760, -1.6), fontsize=5.6,
             color="#0072B2", ha="left", va="bottom")
ax2.set_ylabel("stop-line dist. (m)", fontsize=7)
ax2.set_xlabel("simulation step", fontsize=7)
ax2.tick_params(labelsize=6.5)
ax2.set_xlim(0, 2700)

fig.subplots_adjust(left=0.155, right=0.985, top=0.995, bottom=0.09)
save_fig(fig, "fig_stop")
