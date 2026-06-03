# -*- coding: utf-8 -*-
"""
多模态融合抑郁二分类：以下五部分缺一不可（key 均为 <subject>_<wavstem>）：

1) pause / energy / tremor 文本特征：--feature-root 下须有 pause/、energy/、tremor/ 子目录
2) xvector_keyed.npz：keys + xvector
3) sta_keyed.npz：keys + sta

拼接顺序（每条样本一行向量）：[ xvector | sta | energy统计 | pause | tremor统计 ]。

仅生成融合特征、不训练：加 --export-fused-only，默认写出 fused_five_modalities.npz。

默认路径相对本脚本所在目录 Data_preprocessed_EEGSpeech/。
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FEATURE_ROOT = os.path.join(_SCRIPT_DIR, "pause_energy_tremor", "data")
_DEFAULT_XVECTOR_NPZ = os.path.join(_SCRIPT_DIR, "xvector_ckpt", "xvector_keyed.npz")
_DEFAULT_STA_NPZ = os.path.join(_SCRIPT_DIR, "sta_keyed.npz")
_DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "fusion_depression_best.pth")
# 五路拼接后的单一特征矩阵（顺序：xvector | sta | energy | pause | tremor）
_DEFAULT_FUSED_NPZ = os.path.join(_SCRIPT_DIR, "fused_five_modalities.npz")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def depression_label_from_subject(subject: str) -> int:
    subject = str(subject)
    if subject.startswith("0201"):
        return 1
    if subject.startswith("0202") or subject.startswith("0203"):
        return 0
    raise ValueError(f"Unexpected subject id: {subject!r}")


def parse_key_to_subject_wavstem(key: str) -> Tuple[str, str]:
    # key format: <subject>_<wavstem>
    key = str(key)
    if "_" not in key:
        raise ValueError(f"Invalid key (no underscore): {key!r}")
    subject, wavstem = key.split("_", 1)
    return subject, wavstem


def list_feature_keys(feature_dir: str) -> List[str]:
    keys: List[str] = []
    for fn in os.listdir(feature_dir):
        if not fn.endswith(".txt"):
            continue
        if fn.startswith("skipped_files"):
            continue
        keys.append(os.path.splitext(fn)[0])
    keys.sort()
    return keys


def compute_stats_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(7, dtype=np.float32)

    # robust: replace nan/inf
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(7, dtype=np.float32)

    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)
    stats = np.array(
        [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.median(x)),
            float(p25),
            float(p75),
        ],
        dtype=np.float32,
    )
    return stats


def load_pause_vector(path: str) -> np.ndarray:
    # pause: saved by np.savetxt(data_pause_mul) where data_pause_mul has shape (6,)
    v = np.loadtxt(path, dtype=np.float32)
    v = np.asarray(v).reshape(-1)
    if v.size < 6:
        v = np.pad(v, (0, 6 - v.size), mode="constant")
    elif v.size > 6:
        v = v[:6]
    return v.astype(np.float32, copy=False)


def load_energy_or_tremor_stats(path: str) -> np.ndarray:
    # energy/tremor: saved by np.savetxt on a (2, T) array => two text lines.
    # We convert to fixed-length stats per row.
    arr = np.loadtxt(path, dtype=np.float32)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # If it somehow became 1D, treat it as a single row and pad.
        row0 = arr
        row1 = np.zeros_like(arr)
    else:
        if arr.shape[0] < 2:
            row0 = arr[0]
            row1 = np.zeros_like(row0)
        else:
            row0 = arr[0]
            row1 = arr[1]
    s0 = compute_stats_1d(row0)
    s1 = compute_stats_1d(row1)
    return np.concatenate([s0, s1], axis=0)  # 14 dims


def stratified_subject_split(
    keys: Sequence[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    # Split by subject id to avoid leakage.
    subject_to_keys: Dict[str, List[str]] = {}
    for k in keys:
        subject, _ = parse_key_to_subject_wavstem(k)
        subject_to_keys.setdefault(subject, []).append(k)

    subjects = sorted(subject_to_keys.keys())
    by_label: Dict[int, List[str]] = {0: [], 1: []}
    for s in subjects:
        y = depression_label_from_subject(s)
        by_label[y].append(s)

    rng = np.random.default_rng(seed)

    train_subjects: List[str] = []
    val_subjects: List[str] = []

    for y in [0, 1]:
        sids = by_label[y]
        if len(sids) < 2:
            raise ValueError(f"Label {y} has too few subjects ({len(sids)}). Need >=2.")
        sids = list(sids)
        rng.shuffle(sids)
        n_val = max(int(round(len(sids) * val_ratio)), 1)
        n_val = min(n_val, len(sids) - 1)
        val_subjects.extend(sids[:n_val])
        train_subjects.extend(sids[n_val:])

    rng.shuffle(train_subjects)
    rng.shuffle(val_subjects)

    train_keys: List[str] = []
    val_keys: List[str] = []
    for k in keys:
        subject, _ = parse_key_to_subject_wavstem(k)
        if subject in set(val_subjects):
            val_keys.append(k)
        else:
            train_keys.append(k)
    return train_keys, val_keys


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (logits > 0).long()
    return float((pred.view(-1) == y.view(-1)).float().mean().item())


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def _branch_slices_from_dim_debug(dim_dbg: Dict[str, int]) -> List[Tuple[str, slice]]:
    """
    Feature order:
      [ xvector | sta | energy | pause | tremor ]
    Returns list of (display_name, slice) with indices into the fused feature vector.
    """
    xv = int(dim_dbg.get("xvector_dim", 0))
    sta = int(dim_dbg.get("sta_dim", 0))
    energy = int(dim_dbg.get("energy_dim", 14))
    pause = int(dim_dbg.get("pause_dim", 6))
    tremor = int(dim_dbg.get("tremor_dim", 14))
    total = int(dim_dbg.get("total_dim", xv + sta + energy + pause + tremor))
    cur = 0
    s_xv = slice(cur, cur + xv)
    cur += xv
    s_sta = slice(cur, cur + sta)
    cur += sta
    s_energy = slice(cur, cur + energy)
    cur += energy
    s_pause = slice(cur, cur + pause)
    cur += pause
    s_tremor = slice(cur, cur + tremor)
    cur += tremor
    if cur != total:
        raise ValueError(f"Dim mismatch when building branch slices: computed={cur} vs total_dim={total}")
    return [
        ("x-vector", s_xv),
        ("STA", s_sta),
        ("Energy", s_energy),
        ("Pause", s_pause),
        ("Tremor", s_tremor),
    ]


@torch.no_grad()
def compute_five_branch_occlusion_sensitivity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    branch_slices: List[Tuple[str, slice]],
) -> Dict[str, float]:
    """
    Occlusion sensitivity per branch:
      set that branch's input features to 0, compute absolute change in target probability.

    target probability definition:
      y=1 -> p
      y=0 -> 1-p
    """
    model.eval()
    sums = {name: 0.0 for name, _ in branch_slices}
    n_total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).view(-1)
        bsz = int(yb.numel())
        if bsz == 0:
            continue

        logits_full = model(xb)
        p_full = torch.sigmoid(logits_full)
        p_t_full = torch.where(yb > 0.5, p_full, 1.0 - p_full)

        for name, sl in branch_slices:
            x_occ = xb.clone()
            if sl.stop > sl.start:  # allow 0-dim branches safely
                x_occ[:, sl] = 0.0
            p_occ = torch.sigmoid(model(x_occ))
            p_t_occ = torch.where(yb > 0.5, p_occ, 1.0 - p_occ)
            delta = torch.abs(p_t_full - p_t_occ).sum().item()
            sums[name] += float(delta)

        n_total += bsz

    out = {}
    denom = float(max(n_total, 1))
    for name, _ in branch_slices:
        out[name] = sums[name] / denom
    return out


def save_five_branch_occlusion_barplot(
    sens: Dict[str, float],
    out_path: str,
    *,
    title: str = "Occlusion sensitivity of five speech acoustic branches",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    order = ["x-vector", "STA", "Energy", "Pause", "Tremor"]
    vals = [float(sens.get(k, 0.0)) for k in order]

    fig = plt.figure(figsize=(9.2, 5.2), facecolor="white")
    ax = fig.add_subplot(111)
    bars = ax.bar(np.arange(len(order)), vals, color="#1f77b4", alpha=0.88)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Mean absolute prediction change after occlusion")
    ax.set_title(title)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], padding=3, fontsize=11)
    vmax = float(max(vals) if vals else 0.0)
    if vmax > 0:
        ax.set_ylim(0.0, vmax * 1.25)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, axis="y")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """
    PCA to 2D using SVD (no sklearn dependency).
    x: (N, D)
    returns: (N, 2)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    # U: (N, r), S: (r,), Vt: (r, D)
    u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    r = int(min(2, u.shape[1], s.shape[0]))
    if r <= 0:
        return np.zeros((x.shape[0], 2), dtype=np.float64)
    z = u[:, :r] * s[:r]
    if r == 1:
        z = np.concatenate([z, np.zeros((z.shape[0], 1), dtype=np.float64)], axis=1)
    return z.astype(np.float64, copy=False)


def save_five_branch_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    out_path: str,
    *,
    title: str = "(a) Five-branch PCA distribution",
) -> None:
    """
    Each point = one sample (one <subject>_<wavstem> key under current pipeline granularity).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y shape mismatch: X={X.shape}, y={y.shape}")

    z = _pca_2d(X)
    hc = np.where(y == 0)[0]
    mdd = np.where(y == 1)[0]

    fig = plt.figure(figsize=(7.2, 6.0), facecolor="white")
    ax = fig.add_subplot(111)
    if hc.size > 0:
        ax.scatter(z[hc, 0], z[hc, 1], s=22, c="#1f77b4", alpha=0.75, label="HC", edgecolors="none")
    if mdd.size > 0:
        ax.scatter(z[mdd, 0], z[mdd, 1], s=22, c="#d62728", alpha=0.75, label="MDD", edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_xvector_sta_branch_pca_figure(
    X_fused: np.ndarray,
    y: np.ndarray,
    branch_slices: List[Tuple[str, slice]],
    out_path: str,
) -> None:
    """
    Two-panel PCA scatter:
      (b) x-vector branch
      (c) STA branch

    Each point = one sample (current script granularity: one <subject>_<wavstem> key).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    X_fused = np.asarray(X_fused, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if X_fused.ndim != 2 or X_fused.shape[0] != y.shape[0]:
        raise ValueError(f"X/y shape mismatch: X={X_fused.shape}, y={y.shape}")

    sl_map = {name: sl for name, sl in branch_slices}
    if "x-vector" not in sl_map or "STA" not in sl_map:
        raise ValueError("branch_slices must contain x-vector and STA")

    X_xv = (
        X_fused[:, sl_map["x-vector"]]
        if sl_map["x-vector"].stop > sl_map["x-vector"].start
        else np.zeros((X_fused.shape[0], 1), dtype=np.float32)
    )
    X_sta = (
        X_fused[:, sl_map["STA"]]
        if sl_map["STA"].stop > sl_map["STA"].start
        else np.zeros((X_fused.shape[0], 1), dtype=np.float32)
    )
    z_xv = _pca_2d(X_xv)
    z_sta = _pca_2d(X_sta)

    hc = np.where(y == 0)[0]
    mdd = np.where(y == 1)[0]

    fig = plt.figure(figsize=(13.2, 5.7), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.18)
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])

    def _scatter(ax, z, title):
        if hc.size > 0:
            ax.scatter(z[hc, 0], z[hc, 1], s=20, c="#1f77b4", alpha=0.75, label="HC", edgecolors="none")
        if mdd.size > 0:
            ax.scatter(z[mdd, 0], z[mdd, 1], s=20, c="#d62728", alpha=0.75, label="MDD", edgecolors="none")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    _scatter(ax2, z_xv, "(b) x-vector branch")
    _scatter(ax3, z_sta, "(c) STA branch")
    ax3.legend(loc="upper right", frameon=True, fontsize=10)

    # Give suptitle enough separation from subplot titles
    fig.suptitle("PCA visualizations of x-vector and STA branches", fontsize=16, y=0.985)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def load_keyed_embedding_npz(path: str, array_name: str) -> Dict[str, np.ndarray]:
    """
    Expected npz:
      - keys: (N,) array of strings
      - <array_name>: (N, D) float32/float64
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ck = np.load(path, allow_pickle=True)
    if "keys" not in ck.files or array_name not in ck.files:
        raise ValueError(f"{path} must contain arrays: keys + {array_name}")

    keys = ck["keys"]
    emb = ck[array_name]
    keys = [str(k) for k in keys.tolist()]
    if emb.shape[0] != len(keys):
        raise ValueError(f"Key count mismatch: {len(keys)} vs {emb.shape[0]}")
    out: Dict[str, np.ndarray] = {}
    for i, k in enumerate(keys):
        out[k] = np.asarray(emb[i], dtype=np.float32)
    return out


def build_feature_matrix(
    feature_root: str,
    keys: Sequence[str],
    include_xvector: Optional[str] = None,
    include_sta: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    energy_dir = os.path.join(feature_root, "energy")
    pause_dir = os.path.join(feature_root, "pause")
    tremor_dir = os.path.join(feature_root, "tremor")

    xvec_map: Optional[Dict[str, np.ndarray]] = None
    if include_xvector:
        xvec_map = load_keyed_embedding_npz(include_xvector, "xvector")

    sta_map: Optional[Dict[str, np.ndarray]] = None
    if include_sta:
        sta_map = load_keyed_embedding_npz(include_sta, "sta")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    dim_debug: Dict[str, int] = {}

    for k in keys:
        subject, _ = parse_key_to_subject_wavstem(k)
        y = depression_label_from_subject(subject)

        pause_path = os.path.join(pause_dir, f"{k}.txt")
        energy_path = os.path.join(energy_dir, f"{k}.txt")
        tremor_path = os.path.join(tremor_dir, f"{k}.txt")

        if not os.path.isfile(pause_path) or not os.path.isfile(energy_path) or not os.path.isfile(tremor_path):
            continue

        p = load_pause_vector(pause_path)  # (6,)
        e = load_energy_or_tremor_stats(energy_path)  # (14,)
        t = load_energy_or_tremor_stats(tremor_path)  # (14,)

        # Fusion order (per-sample feature):
        # [xvector, sta, energy, pause, tremor]
        parts: List[np.ndarray] = []

        if xvec_map is not None:
            if k not in xvec_map:
                raise KeyError(f"Missing xvector embedding for key: {k}")
            xv = xvec_map[k]
            parts.append(xv)

        if sta_map is not None:
            if k not in sta_map:
                raise KeyError(f"Missing STA embedding for key: {k}")
            sv = sta_map[k]
            parts.append(sv)

        parts.extend([e, p, t])

        feat = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if feat.ndim != 1:
            raise ValueError(f"Feature must be 1D, got shape: {feat.shape} for key={k!r}")

        if not dim_debug:
            dim_debug = {
                "xvector_dim": int(xv.shape[0]) if xvec_map is not None else 0,
                "sta_dim": int(sv.shape[0]) if sta_map is not None else 0,
                "energy_dim": int(e.shape[0]),
                "pause_dim": int(p.shape[0]),
                "tremor_dim": int(t.shape[0]),
                "total_dim": int(feat.shape[0]),
            }

        X_list.append(feat)
        y_list.append(int(y))

    if not X_list:
        raise RuntimeError("No valid samples found. Check feature_root and filenames.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    dim_debug["n_samples"] = int(X.shape[0])
    dim_debug["input_dim"] = int(X.shape[1])
    return X, y, dim_debug


def standardize_train_val(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-8)
    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std
    return X_train_s, X_val_s, mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Fuse pause+energy+tremor features and detect depression.")
    p.add_argument(
        "--feature-root",
        type=str,
        default=_DEFAULT_FEATURE_ROOT,
        help=f"含 pause/、energy/、tremor/ 子目录（默认: {_DEFAULT_FEATURE_ROOT}）",
    )
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max-samples", type=int, default=None, help="Only use a subset of keys (debug).")
    p.add_argument("--out", type=str, default=_DEFAULT_OUT, help=f"默认: {_DEFAULT_OUT}")

    # 与 pause/energy/tremor 一起，共五项特征来源（均为必填文件）
    p.add_argument(
        "--xvector-npz",
        type=str,
        default=_DEFAULT_XVECTOR_NPZ,
        help=f"npz：keys + xvector（默认: {_DEFAULT_XVECTOR_NPZ}）",
    )
    p.add_argument(
        "--sta-npz",
        type=str,
        default=_DEFAULT_STA_NPZ,
        help=f"npz：keys + sta（默认: {_DEFAULT_STA_NPZ}）",
    )
    p.add_argument(
        "--export-fused-only",
        action="store_true",
        help=f"只把五路拼成新特征并保存 npz，不训练分类器（默认输出: {_DEFAULT_FUSED_NPZ}）",
    )
    p.add_argument(
        "--save-fused-npz",
        type=str,
        default=None,
        help="融合特征输出路径；与训练联用时若指定则额外保存；与 --export-fused-only 联用时指定输出文件",
    )
    args = p.parse_args()

    if not args.xvector_npz:
        raise ValueError("--xvector-npz is required.")
    if not args.sta_npz:
        raise ValueError("--sta-npz is required.")
    if not os.path.isfile(args.xvector_npz):
        raise FileNotFoundError(f"xvector npz not found: {args.xvector_npz}")
    if not os.path.isfile(args.sta_npz):
        raise FileNotFoundError(f"sta npz not found: {args.sta_npz}")

    # 1) list keys from pause (it should exist for all samples we want)
    pause_dir = os.path.join(args.feature_root, "pause")
    if not os.path.isdir(pause_dir):
        raise FileNotFoundError(f"pause dir not found: {pause_dir}")
    keys = list_feature_keys(pause_dir)

    # 2) require energy & tremor exist too (to keep feature vector consistent)
    energy_dir = os.path.join(args.feature_root, "energy")
    tremor_dir = os.path.join(args.feature_root, "tremor")
    valid_keys: List[str] = []
    for k in keys:
        if os.path.isfile(os.path.join(energy_dir, f"{k}.txt")) and os.path.isfile(os.path.join(tremor_dir, f"{k}.txt")):
            valid_keys.append(k)
    if not valid_keys:
        raise RuntimeError("No valid samples (need pause+energy+tremor).")
    keys = valid_keys
    if args.max_samples is not None and args.max_samples > 0 and len(keys) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(keys), size=args.max_samples, replace=False)
        keys = [keys[i] for i in idx.tolist()]

    # 仅导出五路融合后的新特征（npz：keys, feat, label）
    if args.export_fused_only:
        out_npz = args.save_fused_npz or _DEFAULT_FUSED_NPZ
        X_all, y_all, dim_dbg = build_feature_matrix(
            args.feature_root, keys, include_xvector=args.xvector_npz, include_sta=args.sta_npz
        )
        os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
        np.savez(
            out_npz,
            keys=np.array(keys, dtype=object),
            feat=X_all.astype(np.float32),
            label=y_all.astype(np.int64),
        )
        print(f"[export-fused-only] saved: {out_npz}")
        print(f"[export-fused-only] n={X_all.shape[0]} dim={X_all.shape[1]} | block dims: {dim_dbg}")
        return

    # 3) split by subject
    train_keys, val_keys = stratified_subject_split(keys, val_ratio=args.val_ratio, seed=args.seed)

    # 4) build matrices (precompute features once)
    X_train, y_train, dim_train = build_feature_matrix(
        args.feature_root, train_keys, include_xvector=args.xvector_npz, include_sta=args.sta_npz
    )
    X_val, y_val, _ = build_feature_matrix(
        args.feature_root, val_keys, include_xvector=args.xvector_npz, include_sta=args.sta_npz
    )

    # 训练同时可选：额外保存全量 keys 的融合特征
    if args.save_fused_npz:
        X_all, y_all, _ = build_feature_matrix(
            args.feature_root, keys, include_xvector=args.xvector_npz, include_sta=args.sta_npz
        )
        os.makedirs(os.path.dirname(args.save_fused_npz) or ".", exist_ok=True)
        np.savez(
            args.save_fused_npz,
            keys=np.array(keys, dtype=object),
            feat=X_all.astype(np.float32),
            label=y_all.astype(np.int64),
        )
        print(f"[save-fused-npz] saved: {args.save_fused_npz}")

    input_dim = X_train.shape[1]
    X_train_s, X_val_s, mean, std = standardize_train_val(X_train, X_val)

    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 5) model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionMLP(input_dim=input_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # binary logits
    # BCEWithLogitsLoss expects targets in {0,1}
    # (we stored y as float already)
    pos_count = float((y_train == 1).sum())
    neg_count = float((y_train == 0).sum())
    if pos_count > 0 and neg_count > 0:
        # set pos_weight to upweight the minority class
        pos_weight = torch.tensor([neg_count / pos_count], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    best_val_acc = -1.0
    best_state: Dict[str, object] = {}

    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = int(yb.numel())
            running_loss += float(loss.item()) * bs
            n_seen += bs

        train_loss = running_loss / max(n_seen, 1)

        model.eval()
        all_logits: List[torch.Tensor] = []
        all_y: List[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.append(logits.detach().cpu())
                all_y.append(yb.detach().cpu())
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0).long()
        val_acc = accuracy_from_logits(logits, y)

        print(f"[ep {ep:03d}] train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model": model.state_dict(),
                "input_dim": int(input_dim),
                "best_val_acc": float(best_val_acc),
                "feature_mean": mean,
                "feature_std": std,
                "config": vars(args),
                "dim_train_debug": dim_train,
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(best_state, args.out)
    print(f"Saved best checkpoint: {args.out} (best_val_acc={best_val_acc:.4f})")

    # ===== Five-branch occlusion sensitivity (bar plot) =====
    try:
        model.load_state_dict(best_state["model"])
        branch_slices = _branch_slices_from_dim_debug(dim_train)
        sens = compute_five_branch_occlusion_sensitivity(model, val_loader, device, branch_slices)
        out_dir = os.path.dirname(args.out) or "."
        out_png = os.path.join(out_dir, "five_branch_occlusion_sensitivity_bar.png")
        save_five_branch_occlusion_barplot(
            sens,
            out_png,
            title="(a) Five-branch speech occlusion sensitivity",
        )
        print(f"Saved five-branch occlusion sensitivity bar plot: {out_png}")
    except Exception as e:
        print(f"[warn] failed to save five-branch occlusion sensitivity bar plot ({e})")

    # ===== PCA scatter (PC1/PC2): x-vector + STA =====
    try:
        out_dir = os.path.dirname(args.out) or "."
        out_png = os.path.join(out_dir, "pca_xvector_sta_branches_pc1_pc2.png")
        branch_slices = _branch_slices_from_dim_debug(dim_train)
        save_xvector_sta_branch_pca_figure(X_val_s, y_val, branch_slices, out_png)
        print(f"Saved x-vector/STA branch PCA figure: {out_png}")
    except Exception as e:
        print(f"[warn] failed to save x-vector/STA branch PCA figure ({e})")


if __name__ == "__main__":
    main()

