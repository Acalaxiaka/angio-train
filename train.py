# -*- coding: utf-8 -*-
"""冠脉造影投照体位分类训练（第 2 步）。完全离线，数据不离开本机。
用法：python train.py -d D:\\frames -o D:\\runs\\exp1   详见 README。"""
import os, sys, json, time, argparse, shutil, html
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from common import *


def get_args(argv=None):
    ap = argparse.ArgumentParser(description="angio-train 体位分类训练")
    ap.add_argument("-d", "--data", required=True, help="angio-frames 输出文件夹（含 labels.csv 与 images/）")
    ap.add_argument("-o", "--out", required=True, help="训练输出文件夹")
    ap.add_argument("--arch", default="efficientnet_b0", help="efficientnet_b0（默认）/ resnet50 / 其它 timm 名称")
    ap.add_argument("--weights", default=None, help="预训练权重路径（默认 weights/<arch>.pth）")
    ap.add_argument("--no-pretrained", action="store_true", help="不用预训练（不推荐）")
    ap.add_argument("--img", type=int, default=384); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--split", default="0.7,0.15,0.15")
    ap.add_argument("--keep-unknown", action="store_true", help="保留 UNKNOWN 类（默认丢弃）")
    ap.add_argument("--min-count", type=int, default=20, help="类别最少序列数，少于则丢弃")
    ap.add_argument("--multitask", action="store_true", help="增加左右冠 side 第二输出头")
    ap.add_argument("--balance", default="sampler", choices=["sampler", "loss", "none"], help="类别均衡：均衡采样/加权损失/不处理")
    ap.add_argument("--cache", action="store_true", help="把图片缓存到内存（64GB 内存建议开启）")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader 进程数（Windows 默认 4，缓存模式 2）")
    ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--resume", action="store_true", help="从输出文件夹 last.pt 续训")
    ap.add_argument("--max-steps", type=int, default=0, help="调试用：每轮最多步数")
    return ap.parse_args(argv)


def run_epoch(model, loader, dev, amp, crit, crit_side=None, opt=None, scaler=None, max_steps=0):
    train = opt is not None; model.train(train)
    tot, n, P, Y, PS, YS = 0.0, 0, [], [], [], []
    for step, (x, y, s) in enumerate(loader):
        if max_steps and step >= max_steps: break
        x, y, s = x.to(dev, non_blocking=True), y.to(dev), s.to(dev)
        with torch.set_grad_enabled(train), torch.autocast(device_type=dev.type, dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16, enabled=amp):
            out = model(x)
            if isinstance(out, tuple):
                ov, os_ = out; loss = crit(ov, y)
                if (s >= 0).any(): loss = loss + 0.5 * crit_side(os_, s)
                PS.append(os_.float().softmax(1).detach().cpu()); YS.append(s.cpu())
            else:
                ov = out; loss = crit(ov, y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item() * len(y); n += len(y)
        P.append(ov.float().softmax(1).detach().cpu()); Y.append(y.cpu())
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
    r = {"loss": tot / max(n, 1), "acc": float((P.argmax(1) == Y).mean()), "probs": P, "y": Y}
    if PS:
        PS = torch.cat(PS).numpy(); YS = torch.cat(YS).numpy(); m = YS >= 0
        r["side_acc"] = float((PS[m].argmax(1) == YS[m]).mean()) if m.any() else None; r["side_probs"] = PS
    return r


def macro_f1(y, p, k):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(k)), average="macro", zero_division=0))


def series_vote(df, probs):
    """序列级投票：同一序列各帧概率取平均。"""
    d = df[["patient_id", "series"]].copy().reset_index(drop=True)
    key = d["patient_id"].astype(str) + "|" + d["series"].astype(str)
    out = {}
    for k, idx in key.groupby(key).groups.items():
        out[k] = probs[list(idx)].mean(0)
    return key, out


def save_reports(out, classes, df_test, res, multitask):
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    y, p = res["y"], res["probs"].argmax(1); k = len(classes)
    rep = classification_report(y, p, labels=list(range(k)), target_names=classes, zero_division=0, output_dict=True)
    txt = classification_report(y, p, labels=list(range(k)), target_names=classes, zero_division=0, digits=3)
    # 序列级
    key, sv = series_vote(df_test, res["probs"])
    ys = {kk: yy for kk, yy in zip(key, y)}
    s_true = np.array([ys[kk] for kk in sv]); s_pred = np.array([v.argmax() for v in sv.values()])
    s_acc = float((s_true == s_pred).mean()) if len(sv) else None
    open(os.path.join(out, "classification_report.txt"), "w", encoding="utf-8").write(
        "测试集 帧级 分类报告\n" + txt + f"\n序列级投票准确率: {s_acc:.4f} （{len(sv)} 个序列）\n")
    pd.DataFrame(rep).T.to_csv(os.path.join(out, "per_class_report.csv"), encoding="utf-8-sig")
    for name, (yt, yp) in {"confusion_matrix.png": (y, p), "confusion_matrix_series.png": (s_true, s_pred)}.items():
        cm = confusion_matrix(yt, yp, labels=list(range(k)))
        fig, ax = plt.subplots(figsize=(1 + 0.8 * k, 1 + 0.7 * k)); ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(k)); ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticks(range(k)); ax.set_yticklabels(classes)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name.replace(".png", ""))
        fig.tight_layout(); fig.savefig(os.path.join(out, name), dpi=120); plt.close(fig)
    # 误分类画廊
    rows = []
    for i in np.where(y != p)[0]:
        r = df_test.iloc[i]; src = r["full_path"]
        rel = os.path.relpath(src, out).replace("\\", "/") if os.path.splitdrive(src)[0] == os.path.splitdrive(os.path.abspath(out))[0] else "file:///" + src.replace("\\", "/")
        rows.append(f'<div class="c"><img src="{html.escape(rel)}" loading="lazy"><br><b>真:{classes[y[i]]}</b> → <span>预测:{classes[p[i]]}</span> {res["probs"][i, p[i]]:.2f}<br>'
                    f'{html.escape(str(r["patient_id"]))} / {html.escape(str(r["series"]))} f{r["frame_index"]}<br>角度 {r["primary_angle"]}, {r["secondary_angle"]} | {html.escape(str(r["side"]))}</div>')
    open(os.path.join(out, "misclassified.html"), "w", encoding="utf-8").write(
        '<!doctype html><meta charset="utf-8"><title>误分类画廊</title><style>body{font-family:"Microsoft YaHei",sans-serif;background:#222;color:#eee}'
        '.c{display:inline-block;width:260px;margin:6px;font-size:12px;vertical-align:top}.c img{width:256px}span{color:#f88}</style>'
        f'<h2>测试集误分类 {len(rows)} / {len(y)} 帧</h2><p>请复核：标签错误（角度边界/记录错）还是模型错误。</p>' + "".join(rows))
    m = {"test_frame_acc": float((y == p).mean()), "test_frame_macro_f1": macro_f1(y, p, k), "test_series_acc": s_acc,
         "test_series_macro_f1": macro_f1(s_true, s_pred, k) if len(sv) else None, "n_test_frames": int(len(y)), "n_test_series": len(sv),
         "per_class": {c: {kk: rep[c][kk] for kk in ("precision", "recall", "f1-score", "support")} for c in classes}}
    if multitask and res.get("side_acc") is not None: m["test_side_acc"] = res["side_acc"]
    return m


def export_onnx(model, path, img, multitask):
    model = model.float().cpu().eval()
    x = torch.randn(1, 3, img, img)
    names = ["view_logits", "side_logits"] if multitask else ["view_logits"]
    torch.onnx.export(model, x, path, input_names=["input"], output_names=names, opset_version=17,
                      dynamic_axes={"input": {0: "batch"}, **{n: {0: "batch"} for n in names}}, dynamo=False)
    return path


def main(argv=None):
    a = get_args(argv); os.makedirs(a.out, exist_ok=True); seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    amp = (dev.type == "cuda") and not a.no_amp
    print(f"[设备] {dev} {torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''} AMP={amp}")
    if dev.type == "cuda": torch.backends.cudnn.benchmark = True
    df, dropped = load_labels(a.data, drop_unknown=not a.keep_unknown, min_count=a.min_count)
    if dropped: print(f"[丢弃类别] 序列数不足 {a.min_count}: {dropped}")
    if len(df) == 0: sys.exit("没有可用数据，请检查 labels.csv 与 images/ 路径")
    classes = sorted(df["view_label"].unique())
    df["split"] = patient_split(df, tuple(float(t) for t in a.split.split(",")), a.seed)
    df[["path", "patient_id", "series", "view_label", "side", "split"]].to_csv(os.path.join(a.out, "split.csv"), index=False, encoding="utf-8-sig")
    parts = {k: df[df.split == k].reset_index(drop=True) for k in ("train", "val", "test")}
    for k, v in parts.items(): print(f"[{k}] 患者 {v.patient_id.nunique()} 序列 {v.groupby(['patient_id','series']).ngroups} 帧 {len(v)}")
    # Windows 用 spawn 启动子进程，进程过多反而慢；R7 5700X 上 4 个足够喂饱 2060S。缓存模式每个进程会复制一份缓存，故用 2。
    workers = a.workers if a.workers is not None else ((2 if a.cache else 4) if os.name == "nt" else (4 if a.cache else 8))
    ds = {k: AngioDataset(v, classes, a.img, train=(k == "train"), cache=a.cache, multitask=a.multitask) for k, v in parts.items()}
    ytr = np.array(ds["train"].y); counts = np.bincount(ytr, minlength=len(classes)).astype(float)
    sampler, cw = None, None
    if a.balance == "sampler":
        w = 1.0 / np.maximum(counts, 1); sampler = WeightedRandomSampler(torch.tensor(w[ytr], dtype=torch.double), len(ytr), replacement=True)
    elif a.balance == "loss":
        cw = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(classes), dtype=torch.float, device=dev)
    kw = dict(batch_size=a.batch, num_workers=workers, pin_memory=dev.type == "cuda", persistent_workers=workers > 0)
    dl_tr = DataLoader(ds["train"], sampler=sampler, shuffle=sampler is None, drop_last=len(ytr) > a.batch, **kw)
    dl_va = DataLoader(ds["val"], shuffle=False, **kw)
    wpath = None if a.no_pretrained else (a.weights or default_weight_path(a.arch))
    if wpath and not os.path.exists(wpath):
        print(f"[警告] 找不到预训练权重 {wpath}；请先运行 下载权重（setup.bat 已包含）。将从随机初始化训练（效果明显变差）。"); wpath = None
    model = AngioNet(a.arch, len(classes), a.multitask, wpath).to(dev).to(memory_format=torch.channels_last)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05); crit_side = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * max(1, (a.max_steps or len(dl_tr))), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cfg = {"arch": a.arch, "classes": classes, "sides": SIDES, "img": a.img, "multitask": a.multitask, "mean": MEAN, "std": STD, "input": "1x3xHxW 灰度复制3通道, (x/255-mean)/std"}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    start, best, bad, hist = 0, -1.0, 0, []
    last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, bad, hist = ck["epoch"] + 1, ck["best"], ck["bad"], ck["hist"]; print(f"[续训] 从第 {start + 1} 轮继续，当前最佳 {best:.4f}")

    for ep in range(start, a.epochs):
        t0 = time.time()
        orig_step = scaler.step
        def step_and_sched(o):
            orig_step(o)
            if sched.last_epoch < sched.total_steps - 1: sched.step()
        scaler.step = step_and_sched
        tr = run_epoch(model, dl_tr, dev, amp, crit, crit_side, opt, scaler, a.max_steps)
        scaler.step = orig_step
        va = run_epoch(model, dl_va, dev, amp, crit, crit_side, max_steps=a.max_steps)
        f1 = macro_f1(va["y"], va["probs"].argmax(1), len(classes))
        h = {"epoch": ep + 1, "train_loss": tr["loss"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": f1, "sec": round(time.time() - t0, 1)}
        if a.multitask: h["val_side_acc"] = va.get("side_acc")
        hist.append(h); print(json.dumps(h, ensure_ascii=False))
        if f1 > best:
            best, bad = f1, 0; torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep + 1, "val_macro_f1": f1}, os.path.join(a.out, "best.pt"))
            print(f"  ★ 保存最佳模型 val_macro_f1={f1:.4f}")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep, "best": best, "bad": bad, "hist": hist}, last)
        if bad >= a.patience: print(f"[早停] 验证集 {a.patience} 轮未提升"); break
    # 测试
    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=dev, weights_only=False); model.load_state_dict(ck["model"])
    dl_te = DataLoader(ds["test"], shuffle=False, **kw)
    te = run_epoch(model, dl_te, dev, amp, crit, crit_side)
    m = save_reports(a.out, classes, parts["test"], te, a.multitask)
    m.update({"best_val_macro_f1": best, "best_epoch": ck["epoch"], "classes": classes, "dropped_classes": dropped, "history": hist,
              "n_patients": {k: int(v.patient_id.nunique()) for k, v in parts.items()}, "args": vars(a)})
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    p = export_onnx(model, os.path.join(a.out, "model.onnx"), a.img, a.multitask)
    print(f"[完成] 测试集 帧级准确率 {m['test_frame_acc']:.4f} 序列级 {m['test_series_acc']:.4f}；ONNX: {p}")
    print(f"结果文件夹: {os.path.abspath(a.out)}（metrics.json, confusion_matrix.png, classification_report.txt, misclassified.html, model.onnx）")
    return m


if __name__ == "__main__":
    main()
