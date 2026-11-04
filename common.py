# -*- coding: utf-8 -*-
"""angio-train 公共模块：数据读取、患者级划分、数据集、模型。全程离线。"""
import os, json, random
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset

MODEL_ALIASES = {"efficientnet_b0": "efficientnet_b0.ra_in1k", "resnet50": "resnet50.a1_in1k"}
SIDES = ["left", "right"]
MEAN, STD = 0.449, 0.226  # ImageNet 灰度均值/方差


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_labels(data_dir, drop_unknown=True, min_count=20):
    df = pd.read_csv(os.path.join(data_dir, "labels.csv"), encoding="utf-8-sig", dtype={"patient_id": str})
    df["view_label"] = df["view_label"].fillna("UNKNOWN").astype(str)
    df["side"] = df["side"].fillna("unknown").astype(str).str.lower()
    def resolve(p):
        p = str(p).replace("\\", "/")
        for c in (p, os.path.join(data_dir, p), os.path.join(data_dir, "images", p)):
            if os.path.exists(c):
                return c
        return os.path.join(data_dir, p)
    df["full_path"] = [resolve(p) for p in df["path"]]
    df = df[[os.path.exists(p) for p in df["full_path"]]].copy()
    if drop_unknown:
        df = df[df["view_label"] != "UNKNOWN"]
    # 以“序列数”统计类别数量（同序列多帧高度相关）
    cnt = df.groupby("view_label")["series"].nunique()
    keep = cnt[cnt >= min_count].index
    dropped = {k: int(v) for k, v in cnt.items() if k not in keep}
    df = df[df["view_label"].isin(keep)].reset_index(drop=True)
    return df, dropped


def patient_split(df, ratios=(0.70, 0.15, 0.15), seed=42):
    """按患者划分，保证同一患者只出现在一个集合中（无泄漏）。"""
    pids = sorted(df["patient_id"].unique())
    rng = random.Random(seed); rng.shuffle(pids)
    n = len(pids); n_tr = int(round(n * ratios[0])); n_va = int(round(n * ratios[1]))
    if n >= 3:
        n_tr = max(1, min(n_tr, n - 2)); n_va = max(1, min(n_va, n - n_tr - 1))
    sets = {"train": set(pids[:n_tr]), "val": set(pids[n_tr:n_tr + n_va]), "test": set(pids[n_tr + n_va:])}
    split = df["patient_id"].map(lambda p: next(k for k, v in sets.items() if p in v))
    return split


def load_gray(path, size):
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def augment(img_t):
    """X 线造影适用的增强。注意：不做水平/垂直翻转！翻转会把 LAO 变成 RAO 一类的镜像，改变体位语义。"""
    import torchvision.transforms.v2.functional as F
    x = img_t
    if random.random() < 0.8:  # 小角度旋转 + 平移 + 缩放（模拟摆位/放大差异）
        x = F.affine(x, angle=random.uniform(-10, 10), translate=[random.randint(-int(.06 * x.shape[-1]), int(.06 * x.shape[-1])) for _ in range(2)],
                     scale=random.uniform(0.9, 1.1), shear=[0.0])
    if random.random() < 0.8:  # 亮度/对比度/伽马（不同设备窗宽窗位）
        x = (x * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08)).clamp(0, 1)
        x = x.clamp(1e-6, 1) ** random.uniform(0.8, 1.25)
    if random.random() < 0.3:  # 高斯噪声（量子噪声）
        x = (x + torch.randn_like(x) * random.uniform(0.0, 0.03)).clamp(0, 1)
    if random.random() < 0.2:  # 轻度模糊
        x = F.gaussian_blur(x, kernel_size=5, sigma=random.uniform(0.3, 1.2))
    return x


class AngioDataset(Dataset):
    def __init__(self, df, classes, size=384, train=False, cache=False, multitask=False):
        self.paths = df["full_path"].tolist(); self.size = size; self.train = train
        self.y = [classes.index(v) for v in df["view_label"]]
        self.s = [SIDES.index(s) if s in SIDES else -100 for s in df["side"]]  # -100 = 忽略
        self.multitask = multitask
        self.cache = None
        if cache:  # 内存缓存：384² uint8 ≈ 147KB/张，1 万张约 1.5GB
            self.cache = np.stack([load_gray(p, size) for p in self.paths])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = self.cache[i] if self.cache is not None else load_gray(self.paths[i], self.size)
        x = torch.from_numpy(a.copy()).float().div_(255).unsqueeze(0)
        if self.train:
            x = augment(x)
        x = ((x - MEAN) / STD).expand(3, -1, -1).contiguous()
        return x, self.y[i], self.s[i]


class AngioNet(nn.Module):
    def __init__(self, arch, n_classes, multitask=False, pretrained_path=None, try_download=False):
        super().__init__()
        import timm
        name = MODEL_ALIASES.get(arch, arch)
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0)
        if pretrained_path:
            sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if not isinstance(sd, dict) or "state_dict" in sd:
                sd = sd.get("state_dict", sd)
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[权重] 已加载 {pretrained_path}（缺失 {len(missing)}，多余 {len(unexpected)}，分类头多余属正常）")
        elif try_download:
            self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        nf = self.backbone.num_features
        self.drop = nn.Dropout(0.2)
        self.head_view = nn.Linear(nf, n_classes)
        self.head_side = nn.Linear(nf, len(SIDES)) if multitask else None

    def forward(self, x):
        f = self.drop(self.backbone(x))
        v = self.head_view(f)
        if self.head_side is not None:
            return v, self.head_side(f)
        return v


def default_weight_path(arch):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", f"{arch}.pth")
