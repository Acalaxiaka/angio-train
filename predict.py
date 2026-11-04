# -*- coding: utf-8 -*-
"""对新的帧文件夹（PNG，或脱敏 DICOM）预测投照体位，并按序列投票。完全离线。
用法：python predict.py -m D:\\runs\\exp1 -i D:\\new_frames -o D:\\pred"""
import os, sys, re, json, argparse, html
import numpy as np, pandas as pd, torch
from PIL import Image
from common import AngioNet, SIDES

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def dicom_frames(path, n=5):
    import pydicom
    ds = pydicom.dcmread(path, force=True); arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2: arr = arr[None]
    if arr.ndim == 4: arr = arr[..., 0]
    T = arr.shape[0]; idx = np.unique(np.linspace(int(T * .2), max(int(T * .8) - 1, int(T * .2)), n).astype(int))
    out = []
    for i in idx:
        f = arr[i]; lo, hi = np.percentile(f, [0.5, 99.5]); f = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1": f = 1 - f
        out.append((Image.fromarray((f * 255).astype(np.uint8)), int(i)))
    return out


def letterbox(im, size):
    im = im.convert("L"); w, h = im.size; s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    c = Image.new("L", (size, size)); c.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2)); return c


def series_key(path, root):
    rel = os.path.relpath(path, root).replace("\\", "/")
    d, f = os.path.split(rel); stem = os.path.splitext(f)[0]
    return (d + "/" if d else "") + re.sub(r"_f\d+$", "", stem)  # angio-frames 命名：S3_xxxx_f006 → 同序列


def collect(inp, nframes):
    items = []  # (series, name, PIL, frame)
    for r, _, fs in os.walk(inp):
        for f in sorted(fs):
            p = os.path.join(r, f); low = f.lower()
            if low.endswith(IMG_EXT):
                items.append((series_key(p, inp), p, Image.open(p), None))
            elif low.endswith(".dcm") or "." not in f:
                try:
                    for im, i in dicom_frames(p, nframes): items.append((os.path.relpath(p, inp).replace("\\", "/"), p, im, i))
                except Exception as e:
                    print(f"[跳过] {p}: {type(e).__name__}")
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description="angio-train 预测")
    ap.add_argument("-m", "--model", required=True, help="训练输出文件夹（含 best.pt / config.json）或 best.pt 路径")
    ap.add_argument("-i", "--input", required=True); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--frames", type=int, default=5, help="DICOM 每序列抽帧数"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    ckp = a.model if a.model.endswith(".pt") else os.path.join(a.model, "best.pt")
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    ck = torch.load(ckp, map_location="cpu", weights_only=False); cfg = ck["config"]
    model = AngioNet(cfg["arch"], len(cfg["classes"]), cfg["multitask"]); model.load_state_dict(ck["model"]); model.to(dev).eval()
    items = collect(a.input, a.frames)
    if not items: sys.exit("输入文件夹中没有 PNG/DICOM")
    os.makedirs(a.out, exist_ok=True)
    P, PS = [], []
    for b in range(0, len(items), a.batch):
        xs = []
        for _, _, im, _ in items[b:b + a.batch]:
            x = np.asarray(letterbox(im, cfg["img"]), dtype=np.float32) / 255
            xs.append(np.repeat(((x - cfg["mean"]) / cfg["std"])[None], 3, 0))
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            o = model(torch.from_numpy(np.stack(xs)).to(dev))
        if isinstance(o, tuple): PS.append(o[1].float().softmax(1).cpu().numpy()); o = o[0]
        P.append(o.float().softmax(1).cpu().numpy())
    P = np.concatenate(P); PS = np.concatenate(PS) if PS else None; C = cfg["classes"]
    fr = pd.DataFrame({"series": [i[0] for i in items], "file": [i[1] for i in items], "frame": [i[3] for i in items],
                       "pred": [C[k] for k in P.argmax(1)], "conf": P.max(1).round(4)})
    if PS is not None: fr["side_pred"] = [SIDES[k] for k in PS.argmax(1)]
    for j, c in enumerate(C): fr["p_" + c] = P[:, j].round(4)
    fr.to_csv(os.path.join(a.out, "predictions_frames.csv"), index=False, encoding="utf-8-sig")
    rows = []
    for s, g in fr.groupby("series", sort=False):
        pm = P[g.index].mean(0); r = {"series": s, "n_frames": len(g), "view_pred": C[pm.argmax()], "conf": round(float(pm.max()), 4),
                                      "frame_agree": round(float((g["pred"] == C[pm.argmax()]).mean()), 3)}
        if PS is not None: r["side_pred"] = SIDES[PS[g.index].mean(0).argmax()]
        r["needs_review"] = r["conf"] < 0.6 or r["frame_agree"] < 0.6
        rows.append(r)
    sr = pd.DataFrame(rows); sr.to_csv(os.path.join(a.out, "predictions_series.csv"), index=False, encoding="utf-8-sig")
    print(sr.to_string(index=False, max_rows=50))
    print(f"[完成] {len(sr)} 个序列，{int(sr.needs_review.sum())} 个低置信度需复核 → {os.path.abspath(a.out)}")
    return sr


if __name__ == "__main__":
    main()
