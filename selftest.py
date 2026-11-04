# -*- coding: utf-8 -*-
"""自测：合成数据 → 患者级划分无泄漏 → CPU 极短训练 → 报告文件 → ONNX 导出与一致性 → 续训 → 预测。"""
import os, sys, json, shutil, tempfile, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from make_synthetic import make
from common import load_labels, patient_split
import train, predict

results = []
def check(name, fn):
    try: fn(); results.append((name, True)); print(f"[PASS] {name}")
    except BaseException: results.append((name, False)); print(f"[FAIL] {name}"); traceback.print_exc()

tmp = tempfile.mkdtemp(prefix="angio_train_test_"); data = make(os.path.join(tmp, "data")); out = os.path.join(tmp, "run")

def t_split():
    df, _ = load_labels(data, drop_unknown=True, min_count=1)
    assert "UNKNOWN" not in set(df.view_label)
    for seed in (0, 1, 42):
        sp = patient_split(df, seed=seed); sets = {k: set(df.patient_id[sp == k]) for k in ("train", "val", "test")}
        assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"]), "患者泄漏"
        assert all(sets.values())
    assert (patient_split(df, seed=7) == patient_split(df, seed=7)).all(), "划分不可复现"
    n = df.patient_id.nunique(); assert abs(len(sets["train"]) - 0.7 * n) <= 1

def t_min_count():
    df, dropped = load_labels(data, drop_unknown=False, min_count=10)
    assert "UNKNOWN" in dropped and "UNKNOWN" not in set(df.view_label)

args = ["-d", data, "-o", out, "--cpu", "--epochs", "2", "--img", "64", "--batch", "8", "--min-count", "1", "--workers", "0", "--no-pretrained", "--multitask"]
def t_train():
    m = train.main(args)
    for f in ["metrics.json", "confusion_matrix.png", "confusion_matrix_series.png", "classification_report.txt", "per_class_report.csv", "misclassified.html", "best.pt", "last.pt", "model.onnx", "split.csv", "config.json"]:
        assert os.path.exists(os.path.join(out, f)), f
    j = json.load(open(os.path.join(out, "metrics.json"), encoding="utf-8")); assert "test_frame_acc" in j and "test_side_acc" in j

def t_resume():
    m = train.main(args[:6] + ["3"] + args[7:] + ["--resume"])
    j = json.load(open(os.path.join(out, "metrics.json"), encoding="utf-8")); assert len(j["history"]) == 3, len(j["history"])

def t_onnx():
    import onnxruntime as ort, torch
    from common import AngioNet
    ck = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False); c = ck["config"]
    m = AngioNet(c["arch"], len(c["classes"]), c["multitask"]); m.load_state_dict(ck["model"]); m.eval()
    x = np.random.randn(2, 3, 64, 64).astype(np.float32)
    s = ort.InferenceSession(os.path.join(out, "model.onnx"), providers=["CPUExecutionProvider"]); o = s.run(None, {"input": x})
    with torch.no_grad(): t = m(torch.from_numpy(x))
    assert len(o) == 2 and o[0].shape == (2, len(c["classes"])); assert np.allclose(o[0], t[0].numpy(), atol=1e-3)

def t_sampler_loss():
    o2 = os.path.join(tmp, "run2"); train.main(["-d", data, "-o", o2, "--cpu", "--epochs", "1", "--img", "64", "--batch", "8", "--min-count", "1", "--workers", "0", "--no-pretrained", "--balance", "loss", "--cache", "--arch", "resnet18"])
    assert os.path.exists(os.path.join(o2, "model.onnx"))

def t_predict():
    sr = predict.main(["-m", out, "-i", os.path.join(data, "images"), "-o", os.path.join(tmp, "pred"), "--cpu"])
    assert len(sr) == 24 * 4 and "side_pred" in sr

def t_no_hflip():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py"), encoding="utf-8").read()
    assert "hflip" not in src and "RandomHorizontalFlip" not in src

for n, f in [("患者级划分无泄漏/可复现", t_split), ("UNKNOWN 与最少类别过滤", t_min_count), ("CPU 训练+报告(多任务)", t_train),
             ("续训 resume", t_resume), ("ONNX 导出与一致性", t_onnx), ("加权损失+内存缓存+其它骨干", t_sampler_loss),
             ("预测+序列投票", t_predict), ("无水平翻转增强", t_no_hflip)]:
    check(n, f)
shutil.rmtree(tmp, ignore_errors=True)
ok = all(r for _, r in results); print("RESULT:", "ALL PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
