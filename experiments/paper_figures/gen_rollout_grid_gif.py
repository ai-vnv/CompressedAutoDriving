"""Side-by-side rollout GIF for all ten pipeline configurations (A0-A9).

Panels sit on a common absolute step axis, so a configuration whose episode
ends early visibly stops there while the others keep driving. Every frame comes
from the media persisted during the evaluation episode itself; nothing here is
re-rendered.

Two properties of the recorded media shape what this can show, and both are
stated on the panels rather than hidden:

* The media ring buffer keeps only the final 91 steps of an episode, so before
  a configuration's window opens there is nothing to show. Those panels read
  "no frames recorded yet".
* Episodes end at different steps. Once an episode is over, its panel freezes
  on the last recorded frame, dimmed, with the outcome and the ending step.

Curriculum C1, seed 180206, is the one combination captured for all ten
configurations. C1 uses domain randomization, which is why the lane markings
are not the usual white.

Output: assets/rollouts_A0_A9_c1.gif
"""
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageSequence

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
F17 = ART / "f17_optimization_method_order_v1"
F18 = ART / "f18_fp16_control_v1"
FONT_DIR = ROOT / "assets" / "fonts"
OUT = ROOT / "assets" / "rollouts_A0_A9_c1.gif"

CURRICULUM, SEED = "c1", 180206
PANEL_W, PANEL_H = 224, 133
LABEL_H = 30
HEADER_H = 24
COLS, ROWS = 5, 2
STEP_STRIDE = 6          # one output frame per this many simulation steps
FRAME_MS = 150

NAMES = {
    "A0": "original",
    "A1": "prune",
    "A2": "prune + KD(C4)",
    "A3": "prune + KD(bal)",
    "A4": "PTQ only",
    "A5": "prune + PTQ",
    "A6": "KD(bal) + PTQ",
    "A7": "KD(C4)+PTQ+QAT",
    "A8": "KD(bal) + QAT",
    "A9": "KD(bal) + FP16",
}
ORDER = list(NAMES)

BG = (7, 51, 42)
FG = (233, 240, 237)
DIM = (150, 180, 168)
PASS_C = (26, 158, 118)
FAIL_C = (214, 88, 26)
REF_C = (110, 150, 200)


def media_dir(pid):
    if pid == "A9":
        return F18 / "primary_media" / "F16H" / CURRICULUM / f"seed_{SEED}"
    return F17 / "primary_media" / pid / CURRICULUM / f"seed_{SEED}"


def verdict(pid):
    path = (F18 if pid == "A9" else F17) / "results" / "pathway_results.csv"
    key = "F16H" if pid == "A9" else pid
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["pathway_id"] == key and row["curriculum"] == CURRICULUM.upper():
                return row["status"]
    raise KeyError(f"no verdict for {pid} on {CURRICULUM}")


def load_font(name, size):
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


FONT_ID = load_font("Poppins-SemiBold.ttf", 13)
FONT_SM = load_font("Poppins-Regular.ttf", 10)
FONT_BADGE = load_font("Poppins-SemiBold.ttf", 15)


def camera_top(frame):
    """First row of the camera image, below the telemetry banner."""
    px = frame.convert("L").load()
    w, h = frame.size

    def brightness(y):
        return sum(px[x, y] for x in range(0, w, 8)) / (w / 8)

    for y in range(h // 2):
        if all(brightness(y + k) > 170 for k in range(6)):
            return y
    raise RuntimeError("could not locate the camera image under the banner")


def read_panels(pid):
    d = media_dir(pid)
    meta = json.loads((d / "primary_media.json").read_text())
    gif = Image.open(d / "primary_rollout.gif")
    frames, top = [], None
    for f in ImageSequence.Iterator(gif):
        rgb = f.convert("RGB")
        if top is None:
            top = camera_top(rgb)
        frames.append(rgb.crop((0, top, rgb.width, rgb.height))
                      .resize((PANEL_W, PANEL_H), Image.LANCZOS))
    return frames, meta


panels = {pid: read_panels(pid) for pid in ORDER}
verdicts = {pid: verdict(pid) for pid in ORDER}

first_recorded = min(m["frame_index_first"] for _, m in panels.values()
                     if m["episode_length"] > 200)
last_step = max(m["episode_length"] for _, m in panels.values())
steps = list(range(first_recorded, last_step + 1, STEP_STRIDE))

CELL_W, CELL_H = PANEL_W, PANEL_H + LABEL_H
CANVAS = (COLS * CELL_W, ROWS * CELL_H + HEADER_H)
DEAD = Image.new("RGB", (PANEL_W, PANEL_H), (12, 30, 26))


def frame_at(pid, step):
    """(image, state) for one configuration at an absolute simulation step."""
    frames, meta = panels[pid]
    first, last = meta["frame_index_first"], meta["frame_index_last"]
    end = meta["episode_length"] - 1
    if step > end:
        return ImageEnhance.Brightness(frames[-1]).enhance(0.45), "ended"
    if step < first:
        return DEAD, "waiting"
    idx = round((step - first) / max(last - first, 1) * (len(frames) - 1))
    return frames[min(max(idx, 0), len(frames) - 1)], "playing"


def badge(draw, x, y, text, colour):
    pad_x, pad_y = 7, 4
    w = draw.textlength(text, font=FONT_BADGE)
    draw.rectangle([x, y, x + w + 2 * pad_x, y + 15 + 2 * pad_y], fill=colour)
    draw.text((x + pad_x, y + pad_y - 1), text, font=FONT_BADGE,
              fill=(255, 255, 255))


def compose(step):
    canvas = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(canvas)
    for k, pid in enumerate(ORDER):
        col, row = k % COLS, k // COLS
        x, y = col * CELL_W, row * CELL_H + HEADER_H
        _, meta = panels[pid]
        img, state = frame_at(pid, step)
        canvas.paste(img, (x, y))

        status = verdicts[pid]
        colour = {"PASS": PASS_C, "FAIL": FAIL_C}.get(status, REF_C)
        end_step = meta["episode_length"] - 1

        if state == "ended":
            word = {"FAIL": "FAILED", "PASS": "COMPLETED"}.get(status, "COMPLETED")
            badge(draw, x + 8, y + 12, f"{word}  step {end_step}", colour)
        elif state == "waiting":
            draw.text((x + 8, y + PANEL_H // 2 - 6), "no frames recorded yet",
                      font=FONT_SM, fill=DIM)

        draw.rectangle([x, y + PANEL_H, x + CELL_W - 1, y + CELL_H - 1], fill=BG)
        draw.text((x + 6, y + PANEL_H + 3), f"{pid}  {NAMES[pid]}",
                  font=FONT_ID, fill=FG)
        draw.text((x + 6, y + PANEL_H + 18), f"episode ends at {end_step}",
                  font=FONT_SM, fill=DIM)
        note = "reference" if status == "REFERENCE" else status.lower()
        w = draw.textlength(note, font=FONT_SM)
        draw.text((x + CELL_W - w - 7, y + PANEL_H + 18), note,
                  font=FONT_SM, fill=colour)

    draw.rectangle([0, 0, CANVAS[0], HEADER_H - 1], fill=BG)
    left = "A0-A9 on C1, seed 180206 - same scenario, same seed"
    draw.text((8, 5), left, font=FONT_SM, fill=DIM)
    label = f"simulation step {step}"
    w = draw.textlength(label, font=FONT_ID)
    draw.text((CANVAS[0] - w - 10, 4), label, font=FONT_ID, fill=FG)
    return canvas


out_frames = [compose(s) for s in steps]
OUT.parent.mkdir(parents=True, exist_ok=True)
out_frames[0].save(OUT, save_all=True, append_images=out_frames[1:], loop=0,
                   duration=FRAME_MS, optimize=True)
print(f"{OUT.relative_to(ROOT)}: {CANVAS[0]}x{CANVAS[1]}, {len(steps)} frames, "
      f"steps {steps[0]}-{steps[-1]}, {OUT.stat().st_size / 1e6:.2f} MB")
for pid in ORDER:
    _, m = panels[pid]
    print(f"  {pid}: ends {m['episode_length'] - 1}, window "
          f"{m['frame_index_first']}-{m['frame_index_last']}, {verdicts[pid]}")
