# -*- coding: utf-8 -*-
"""一次性下载 ImageNet 预训练权重到 weights/ 文件夹（之后训练完全离线）。
顺序：huggingface.co → hf-mirror.com（国内镜像）。也可手动下载后放入 weights/。"""
import os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="all", help="efficientnet_b0 / resnet50 / all")
ap.add_argument("--mirror-first", action="store_true", help="先用 hf-mirror.com")
a = ap.parse_args()
archs = ["efficientnet_b0", "resnet50"] if a.arch == "all" else [a.arch]
endpoints = ["https://huggingface.co", "https://hf-mirror.com"]
if a.mirror_first: endpoints.reverse()
here = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(here, "weights"), exist_ok=True)
ok = True
for arch in archs:
    out = os.path.join(here, "weights", f"{arch}.pth")
    if os.path.exists(out): print(f"[已存在] {out}"); continue
    done = False
    for ep in endpoints:
        os.environ["HF_ENDPOINT"] = ep
        try:
            import importlib, huggingface_hub.constants as C
            importlib.reload(C)
            import timm, torch
            from common import MODEL_ALIASES
            print(f"[下载] {arch} 来自 {ep} ...")
            m = timm.create_model(MODEL_ALIASES[arch], pretrained=True, num_classes=0)
            torch.save(m.state_dict(), out); print(f"[完成] {out}"); done = True; break
        except Exception as e:
            print(f"[失败] {ep}: {type(e).__name__}: {str(e)[:200]}")
    if not done:
        ok = False
        print(f"""
[手动下载] {arch} 自动下载失败。请在能上网的电脑浏览器打开：
  https://hf-mirror.com/timm/{MODEL_ALIASES[arch]}/tree/main
下载 model.safetensors（或 pytorch_model.bin），复制到本机后运行：
  venv\\Scripts\\python convert_weights.py 下载的文件 weights\\{arch}.pth
或训练时直接指定 --weights 下载的文件路径。""")
sys.exit(0 if ok else 1)
