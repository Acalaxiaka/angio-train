# -*- coding: utf-8 -*-
"""生成合成 angio-frames 输出（仅供自测，不含任何真实数据）。不同体位画不同方向的“血管”线条。"""
import os, sys, random, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFilter
VIEWS = ["RAO_CAU", "LAO_CRA", "AP_CRA", "LAO_CAU"]
ANG = {"RAO_CAU": (-30, -25), "LAO_CRA": (40, 25), "AP_CRA": (0, 30), "LAO_CAU": (45, -30)}


def make(out, n_pat=24, series_per=4, frames=3, size=128, seed=0):
    rng = random.Random(seed); rows = []
    for p in range(1, n_pat + 1):
        pid = f"P{p:04d}"; os.makedirs(os.path.join(out, "images", pid), exist_ok=True)
        for s in range(series_per):
            v = VIEWS[(p + s) % len(VIEWS)] if s < series_per - 1 or p % 5 else "UNKNOWN"
            side = rng.choice(["left", "right", "unknown"]); ang = (VIEWS.index(v) * 45 if v != "UNKNOWN" else 0)
            for f in range(frames):
                im = Image.new("L", (size, size), 140); d = ImageDraw.Draw(im)
                cx, cy = size / 2 + rng.uniform(-8, 8), size / 2 + rng.uniform(-8, 8)
                for k in range(3):
                    t = np.deg2rad(ang + k * 15 + rng.uniform(-5, 5)); L = size * 0.4
                    d.line([(cx, cy), (cx + L * np.cos(t), cy + L * np.sin(t))], fill=30, width=4)
                if side == "right": d.ellipse([5, 5, 25, 25], fill=60)
                im = im.filter(ImageFilter.GaussianBlur(1))
                a = np.asarray(im).astype(np.float32) + np.random.RandomState(p * 100 + s * 10 + f).normal(0, 6, (size, size))
                rel = f"images/{pid}/S{s}_{p:04x}{s:04x}_f{f:03d}.png"
                Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(os.path.join(out, rel))
                pa, sa = ANG.get(v, ("", ""))
                rows.append(dict(path=rel, patient_id=pid, study="1", series=f"S{s}", frame_index=f, primary_angle=pa, secondary_angle=sa,
                                 view_label=v, side=side, series_description="synthetic", fps=15, n_frames=frames, source_file=f"syn/{pid}/{s}"))
    pd.DataFrame(rows).to_csv(os.path.join(out, "labels.csv"), index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
