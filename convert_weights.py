# -*- coding: utf-8 -*-
"""把手动下载的 model.safetensors / .bin 转成 weights/xxx.pth。用法：python convert_weights.py 输入 输出.pth"""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
if src.endswith(".safetensors"):
    from safetensors.torch import load_file; sd = load_file(src)
else:
    sd = torch.load(src, map_location="cpu", weights_only=True)
sd = {k: v for k, v in sd.items() if not k.startswith(("classifier.", "fc.", "head."))}
torch.save(sd, dst); print("已保存", dst, len(sd), "个张量")
