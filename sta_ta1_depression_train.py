# -*- coding: utf-8 -*-
"""
STA_TA1 Speech Depression Binary Classification (log-Mel + TA1)

数据约定（沿用你现有工程标注规则）：
- 原始音频（默认）：Data_preprocessed_EEGSpeech/audio_lanzhou_2015/<subject_id>/xx.wav（52 被试）
- 0201 -> MDD (label=1)
- 0202/0203 -> HC  (label=0)

这个脚本是“新脚本”，不依赖现有 xvector 离线流程。
实现流程（对应你的伪代码）：
1) raw waveform -> log-Mel spectrogram（--mode plot-log-mel 可导出 control/depression 示例图，默认 magma 类常规频谱配色）
2) max-pooling & avg-pooling 聚合 + 下采样
3) MobileNetV4-like 2D backbone 提取高层时频特征
4) TA1 dynamic local attentio、n（为每个 head、每个 token 预测 offset & window size）
5) 三阶段监督：I_S1 / I_S2 / I_Sp 各自分类 + focal loss 加权求和
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_AUDIO_ROOT = os.path.join(_SCRIPT_DIR, "audio_lanzhou_2015")
_DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "checkpoints")


def depression_label_from_subject_folder(folder_name: str) -> int:
    """0201 -> MDD(1); 0202/0203 -> HC(0)."""
    folder_name = str(folder_name)
    if folder_name.startswith("0201"):
        return 1
    if folder_name.startswith("0202") or folder_name.startswith("0203"):
        return 0
    raise ValueError(f"Unexpected subject folder name (need 0201/0202/0203): {folder_name!r}")


def is_subject_audio_folder(folder_name: str) -> bool:
    """仅 0201/0202/0203 开头的被试目录；跳过 transcripts 等杂项文件夹。"""
    try:
        depression_label_from_subject_folder(folder_name)
        return True
    except ValueError:
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_wav_files(audio_root: str) -> List[str]:
    exts = (".wav", ".wave", ".mp3", ".m4a", ".flac")
    paths: List[str] = []
    for sd in sorted(os.listdir(audio_root)):
        sub = os.path.join(audio_root, sd)
        if not os.path.isdir(sub) or not is_subject_audio_folder(sd):
            continue
        for fn in sorted(os.listdir(sub)):
            if fn.lower().endswith(exts):
                paths.append(os.path.normpath(os.path.join(sub, fn)))
    return paths


def subject_id_from_wav_path(wav_path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.normpath(wav_path)))


def stratified_split_subjects(
    audio_root: str, val_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    """
    Split by subject folder to avoid leakage.
    Return (train_subjects, val_subjects).
    """
    subjects = []
    for sd in sorted(os.listdir(audio_root)):
        sub = os.path.join(audio_root, sd)
        if os.path.isdir(sub) and is_subject_audio_folder(sd):
            subjects.append(sd)
    if not subjects:
        raise FileNotFoundError(f"No subject folders (0201/0202/0203) under: {audio_root}")

    by_label = {0: [], 1: []}
    for sd in subjects:
        y = depression_label_from_subject_folder(sd)
        by_label[y].append(sd)

    rng = random.Random(seed)
    train_subjects: List[str] = []
    val_subjects: List[str] = []

    for y in [0, 1]:
        ids = by_label[y]
        if len(ids) < 2:
            raise ValueError(f"Label {y} has too few subjects ({len(ids)}). Need >=2.")
        rng.shuffle(ids)
        n_val = max(int(round(len(ids) * val_ratio)), 1)
        n_val = min(n_val, len(ids) - 1)
        val_subjects.extend(ids[:n_val])
        train_subjects.extend(ids[n_val:])

    rng.shuffle(train_subjects)
    rng.shuffle(val_subjects)
    return train_subjects, val_subjects


def audio_duration_seconds(path: str) -> Optional[float]:
    """
    Robustly probe audio duration.

    Some wav files in your dataset may not be readable by `soundfile` or
    `librosa.get_duration` (missing backend). In that case we return None
    and let caller skip the file instead of crashing.
    """
    # 1) soundfile (fast, common)
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(path)
        if info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    # 2) torchaudio (often more tolerant for wave encodings)
    try:
        import torchaudio  # type: ignore

        info = torchaudio.info(path)
        if getattr(info, "num_frames", 0) > 0 and getattr(info, "sample_rate", 0) > 0:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass

    # 3) librosa fallback (may still fail if audioread backend missing)
    try:
        import librosa

        return float(librosa.get_duration(path=path))
    except Exception:
        return None


def load_wave_segment(
    wav_path: str,
    start_sec: float,
    segment_sec: float,
    target_sr: int,
) -> np.ndarray:
    """
    Load just [start_sec, start_sec + segment_sec] with librosa offset/duration.
    Output: float32 mono waveform of exact length segment_sec * target_sr (zero padded).
    """
    import librosa

    try:
        y, sr = librosa.load(
            wav_path,
            sr=target_sr,
            mono=True,
            offset=float(start_sec),
            duration=float(segment_sec),
        )
    except Exception:
        y = None
        sr = target_sr

    if y is None or getattr(y, "size", 0) == 0:
        return np.zeros(int(round(segment_sec * target_sr)), dtype=np.float32)

    y = y.astype(np.float32)
    target_len = int(round(segment_sec * target_sr))
    if y.shape[0] < target_len:
        pad = target_len - y.shape[0]
        y = np.pad(y, (0, pad), mode="constant")
    elif y.shape[0] > target_len:
        y = y[:target_len]
    return y


def waveform_to_log_mel_sta(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_mels: int,
    win_ms: float,
    hop_ms: float,
    segment_sec: float,
    eps: float = 1e-6,
    apply_per_sample_norm: bool = False,
) -> np.ndarray:
    """
    与本脚本 SpeechSegmentsDataset._log_mel 一致：librosa melspectrogram + log，
    并按 segment 长度对齐帧数（与训练张量形状一致）。
    apply_per_sample_norm=True 时再做 (x-mean)/std，与 __getitem__ 一致。
    返回 (n_mels, T)。
    """
    import librosa

    n_fft = int(round(sample_rate * win_ms / 1000.0))
    hop_length = int(round(sample_rate * hop_ms / 1000.0))
    n_samples = int(round(segment_sec * sample_rate))
    if n_samples < n_fft:
        raise ValueError(f"segment_sec too small for window: {segment_sec}s < {win_ms}ms")
    expected_frames = (n_samples - n_fft) // hop_length + 1

    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=False,
        power=2.0,
        n_mels=n_mels,
        fmin=20.0,
        fmax=sample_rate / 2.0,
    )
    mel = np.maximum(mel, 0.0)
    log_mel = np.log(mel + eps).astype(np.float32)
    t = log_mel.shape[1]
    if t < expected_frames:
        log_mel = np.pad(log_mel, ((0, 0), (0, expected_frames - t)), mode="constant")
    elif t > expected_frames:
        log_mel = log_mel[:, :expected_frames]

    if apply_per_sample_norm:
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
    return log_mel


def save_log_mel_figure(
    log_mel_nm: np.ndarray,
    out_path: str,
    *,
    title: str,
    hop_ms: float,
    cmap: str = "magma",
    dpi: int = 150,
) -> None:
    """log-Mel (n_mels, T) 热图；默认 magma（与常见 librosa 频谱图紫→粉→黄风格一致）。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plot log-Mel 需要 matplotlib（pip install matplotlib）") from e

    x = np.asarray(log_mel_nm, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(x, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap)
    ax.set_xlabel(f"Frame (~{hop_ms:g} ms / frame)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log power")
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def try_save_log_mel_examples_by_depression_label(
    audio_root: str,
    *,
    out_hc: Optional[str] = None,
    out_mdd: Optional[str] = None,
    sample_rate: int,
    n_mels: int,
    win_ms: float,
    hop_ms: float,
    segment_sec: float,
    apply_per_sample_norm: bool = False,
    cmap: str = "magma",
) -> Tuple[bool, bool]:
    """
    与 xvector 流程类似：按扫盘顺序各取首个可读的 control(0202/0203) 与 depression(0201)，
    用本脚本 log-Mel 提取并保存 PNG。返回 (saved_hc, saved_mdd)。
    """
    if not out_hc and not out_mdd:
        return False, False
    paths = list_wav_files(audio_root)
    saved_hc = False
    saved_mdd = False
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            sid = subject_id_from_wav_path(p)
            label = depression_label_from_subject_folder(sid)
        except ValueError:
            continue
        want = (label == 0 and out_hc and not saved_hc) or (label == 1 and out_mdd and not saved_mdd)
        if not want:
            continue
        try:
            wav = load_wave_segment(p, 0.0, segment_sec, sample_rate)
            lm = waveform_to_log_mel_sta(
                wav,
                sample_rate=sample_rate,
                n_mels=n_mels,
                win_ms=win_ms,
                hop_ms=hop_ms,
                segment_sec=segment_sec,
                apply_per_sample_norm=apply_per_sample_norm,
            )
            if label == 0 and out_hc and not saved_hc:
                save_log_mel_figure(
                    lm,
                    out_hc,
                    title="(c) control",
                    hop_ms=hop_ms,
                    cmap=cmap,
                )
                saved_hc = True
            elif label == 1 and out_mdd and not saved_mdd:
                save_log_mel_figure(
                    lm,
                    out_mdd,
                    title="(d) Depression",
                    hop_ms=hop_ms,
                    cmap=cmap,
                )
                saved_mdd = True
        except ImportError:
            raise
        except Exception:
            continue
        done = True
        if out_hc and not saved_hc:
            done = False
        if out_mdd and not saved_mdd:
            done = False
        if done:
            break
    return saved_hc, saved_mdd


@dataclass(frozen=True)
class SegmentRow:
    wav_path: str
    start_sec: float
    label: int


class SpeechSegmentsDataset(Dataset):
    def __init__(
        self,
        segments: Sequence[SegmentRow],
        sample_rate: int,
        n_mels: int,
        win_ms: float,
        hop_ms: float,
        segment_sec: float,
    ):
        self.segments = list(segments)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_ms = win_ms
        self.hop_ms = hop_ms
        self.segment_sec = segment_sec

        self.n_fft = int(round(sample_rate * win_ms / 1000.0))
        self.hop_length = int(round(sample_rate * hop_ms / 1000.0))
        self.win_length = self.n_fft

        # With center=False: frames = floor((N - n_fft)/hop) + 1
        n_samples = int(round(segment_sec * sample_rate))
        if n_samples < self.n_fft:
            raise ValueError(f"segment_sec too small for n_fft: {segment_sec}s < {self.win_ms}ms.")
        self.expected_frames = (n_samples - self.n_fft) // self.hop_length + 1

        self.eps = 1e-6

    def __len__(self) -> int:
        return len(self.segments)

    def _log_mel(self, waveform: np.ndarray) -> torch.Tensor:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            center=False,
            power=2.0,
            n_mels=self.n_mels,
            fmin=20.0,
            fmax=self.sample_rate / 2.0,
        )  # (n_mels, T)
        mel = np.maximum(mel, 0.0)
        log_mel = np.log(mel + self.eps).astype(np.float32)

        # Safety: enforce expected frames.
        t = log_mel.shape[1]
        if t < self.expected_frames:
            pad = self.expected_frames - t
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant")
        elif t > self.expected_frames:
            log_mel = log_mel[:, : self.expected_frames]
        return torch.from_numpy(log_mel)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.segments[idx]
        wav = load_wave_segment(
            row.wav_path,
            start_sec=row.start_sec,
            segment_sec=self.segment_sec,
            target_sr=self.sample_rate,
        )
        x = self._log_mel(wav)  # (n_mels, T)
        # Normalization: per-sample mean/std (helps training stability)
        x = (x - x.mean()) / (x.std() + 1e-5)
        return x, row.label


@dataclass(frozen=True)
class ExtractSegmentRow:
    wav_path: str
    start_sec: float
    label: int
    wav_key: str  # <subject>_<wavstem>


class SpeechSegmentsExtractionDataset(Dataset):
    """
    Like SpeechSegmentsDataset, but also returns wav_key so we can aggregate
    per original wav file (instead of per segment).
    """

    def __init__(
        self,
        segments: Sequence[ExtractSegmentRow],
        sample_rate: int,
        n_mels: int,
        win_ms: float,
        hop_ms: float,
        segment_sec: float,
    ):
        self.segments = list(segments)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_ms = win_ms
        self.hop_ms = hop_ms
        self.segment_sec = segment_sec

        self.n_fft = int(round(sample_rate * win_ms / 1000.0))
        self.hop_length = int(round(sample_rate * hop_ms / 1000.0))
        self.win_length = self.n_fft

        n_samples = int(round(segment_sec * sample_rate))
        if n_samples < self.n_fft:
            raise ValueError(f"segment_sec too small for n_fft: {segment_sec}s < {self.win_ms}ms.")
        self.expected_frames = (n_samples - self.n_fft) // self.hop_length + 1

        self.eps = 1e-6

    def __len__(self) -> int:
        return len(self.segments)

    def _log_mel(self, waveform: np.ndarray) -> torch.Tensor:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            center=False,
            power=2.0,
            n_mels=self.n_mels,
            fmin=20.0,
            fmax=self.sample_rate / 2.0,
        )  # (n_mels, T)
        mel = np.maximum(mel, 0.0)
        log_mel = np.log(mel + self.eps).astype(np.float32)

        t = log_mel.shape[1]
        if t < self.expected_frames:
            pad = self.expected_frames - t
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant")
        elif t > self.expected_frames:
            log_mel = log_mel[:, : self.expected_frames]
        return torch.from_numpy(log_mel)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.segments[idx]
        wav = load_wave_segment(
            row.wav_path,
            start_sec=row.start_sec,
            segment_sec=self.segment_sec,
            target_sr=self.sample_rate,
        )
        x = self._log_mel(wav)
        x = (x - x.mean()) / (x.std() + 1e-5)
        return x, row.label, row.wav_key


def build_segments_from_audio_root(
    audio_root: str,
    subject_list: Sequence[str],
    segment_sec: float,
    stride_sec: float,
    max_segments_per_file: Optional[int] = None,
) -> List[SegmentRow]:
    exts = (".wav", ".wave", ".mp3", ".m4a", ".flac")
    segments: List[SegmentRow] = []
    skipped_files = 0
    for sd in subject_list:
        sub_dir = os.path.join(audio_root, sd)
        if not os.path.isdir(sub_dir):
            continue
        label = depression_label_from_subject_folder(sd)
        for fn in sorted(os.listdir(sub_dir)):
            if not fn.lower().endswith(exts):
                continue
            wav_path = os.path.normpath(os.path.join(sub_dir, fn))
            dur = audio_duration_seconds(wav_path)
            if dur is None:
                # Skip unreadable/broken audio file.
                skipped_files += 1
                continue

            if dur <= segment_sec + 1e-4:
                starts = [0.0]
            else:
                # Cover beginning to end (including last possible window)
                max_start = max(0.0, dur - segment_sec)
                starts = []
                s = 0.0
                while s < max_start - 1e-6:
                    starts.append(s)
                    s += stride_sec
                if not starts or abs(starts[-1] - max_start) > 1e-3:
                    starts.append(max_start)

            if max_segments_per_file is not None:
                starts = starts[: int(max_segments_per_file)]

            for st in starts:
                segments.append(SegmentRow(wav_path=wav_path, start_sec=float(st), label=label))
    if not segments:
        raise RuntimeError("No segments generated. Check audio_root and subject folders.")
    if skipped_files > 0:
        print(f"[build_segments] skipped unreadable audio files: {skipped_files}")
    return segments


def build_extraction_segments_from_audio_root(
    audio_root: str,
    subject_list: Sequence[str],
    segment_sec: float,
    stride_sec: float,
    max_segments_per_file: Optional[int] = None,
) -> List[ExtractSegmentRow]:
    """
    Build segment rows for embedding extraction.
    Each segment keeps a wav_key (<subject>_<wavstem>) so we can later average per wav.
    """
    exts = (".wav", ".wave", ".mp3", ".m4a", ".flac")
    segments: List[ExtractSegmentRow] = []
    skipped_files = 0

    for sd in subject_list:
        sub_dir = os.path.join(audio_root, sd)
        if not os.path.isdir(sub_dir):
            continue
        label = depression_label_from_subject_folder(sd)
        for fn in sorted(os.listdir(sub_dir)):
            if not fn.lower().endswith(exts):
                continue
            wav_path = os.path.normpath(os.path.join(sub_dir, fn))
            dur = audio_duration_seconds(wav_path)
            if dur is None:
                skipped_files += 1
                continue

            wav_stem = os.path.splitext(fn)[0]
            wav_key = f"{sd}_{wav_stem}"

            if dur <= segment_sec + 1e-4:
                starts = [0.0]
            else:
                max_start = max(0.0, dur - segment_sec)
                starts = []
                s = 0.0
                while s < max_start - 1e-6:
                    starts.append(s)
                    s += stride_sec
                if not starts or abs(starts[-1] - max_start) > 1e-3:
                    starts.append(max_start)

            if max_segments_per_file is not None:
                starts = starts[: int(max_segments_per_file)]

            for st in starts:
                segments.append(
                    ExtractSegmentRow(wav_path=wav_path, start_sec=float(st), label=label, wav_key=wav_key)
                )

    if not segments:
        raise RuntimeError("No extraction segments generated. Check audio_root and subject folders.")
    if skipped_files > 0:
        print(f"[build_extraction_segments] skipped unreadable audio files: {skipped_files}")
    return segments


class FocalLoss(nn.Module):
    """
    Multi-class focal loss for logits.
    For binary classification we still use 2-way logits (CrossEntropy style).
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        ce = F.cross_entropy(logits, targets, reduction="none")  # (B,)
        pt = torch.exp(-ce)  # (B,) = prob of true class
        loss = (1.0 - pt).pow(self.gamma) * ce

        if self.alpha is not None:
            at = torch.where(targets == 1, torch.tensor(self.alpha, device=targets.device), torch.tensor(1.0 - self.alpha, device=targets.device))
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class HSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class ConvBNAct2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, groups: int = 1, act: bool = True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act
        self.act_fn = HSwish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.act_fn(x)
        return x


class MBConv2d(nn.Module):
    """
    Inverted residual / depthwise separable block (MobileNet-family).
    Not a strict MobileNetV4 reproduction, but a lightweight MobileNet-like 2D backbone.
    """

    def __init__(self, in_ch: int, out_ch: int, exp_ch: int, stride: int):
        super().__init__()
        self.use_res = stride == 1 and in_ch == out_ch
        self.pw = nn.Conv2d(in_ch, exp_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_ch)
        self.act = HSwish()

        self.dw = nn.Conv2d(
            exp_ch,
            exp_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=exp_ch,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(exp_ch)

        self.pw2 = nn.Conv2d(exp_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dw(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.pw2(out)
        out = self.bn3(out)
        if self.use_res:
            out = out + x
        return out


class MobileNetV4LikeBackbone(nn.Module):
    """
    Produces three feature maps:
    - F1: after stage1
    - F2: after stage2
    - Fp: final feature map for TA1
    """

    def __init__(self, in_ch: int = 1, c1: int = 32, c2: int = 64, cp: int = 96):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct2d(in_ch, c1, k=3, s=1, groups=1, act=True),
            ConvBNAct2d(c1, c1, k=3, s=1, groups=1, act=True),
        )

        # Stage 1 (downsample)
        self.stage1 = nn.Sequential(
            MBConv2d(c1, c1, exp_ch=c1 * 2, stride=2),
            MBConv2d(c1, c1, exp_ch=c1 * 2, stride=1),
        )

        # Stage 2 (downsample)
        self.stage2 = nn.Sequential(
            MBConv2d(c1, c2, exp_ch=c2 * 2, stride=2),
            MBConv2d(c2, c2, exp_ch=c2 * 2, stride=1),
        )

        # Stage p (light refinement)
        self.stagep = nn.Sequential(
            MBConv2d(c2, cp, exp_ch=cp * 2, stride=1),
            MBConv2d(cp, cp, exp_ch=cp * 2, stride=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, 1, F, T)
        F1, F2, Fp: (B, C, F', T')
        """
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        fp = self.stagep(f2)
        return f1, f2, fp


class TA1DynamicLocalAttention(nn.Module):
    """
    Dynamic local attention:
    for each head i and token n:
      predict window size s and offset o relative to token index
      compute attention only inside [L:R] = [anchor - s, anchor + s]
    """

    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        self.qkv_proj = nn.Linear(d_model, inner * 3, bias=True)
        self.decision_s = nn.Linear(head_dim, 1, bias=True)
        self.decision_o = nn.Linear(head_dim, 1, bias=True)

    def forward(self, tokens: torch.Tensor, max_window: int) -> torch.Tensor:
        """
        tokens: (B, T, d_model)
        return: (B, T, num_heads * head_dim)
        """
        b, t, d = tokens.shape
        inner = self.num_heads * self.head_dim
        assert d == self.qkv_proj.in_features

        qkv = self.qkv_proj(tokens)  # (B, T, 3*inner)
        qkv = qkv.view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, T, heads, head_dim)
        # Move heads dimension forward for easier indexing: (B, heads, T, head_dim)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        out = torch.zeros((b, self.num_heads, t, self.head_dim), device=tokens.device, dtype=tokens.dtype)
        scale = float(self.head_dim) ** -0.5

        # Python loops for correctness under dynamic windows.
        # If you need speed-up later, we can switch to a CUDA-friendly implementation.
        max_window = int(max_window)
        for bi in range(b):
            for hi in range(self.num_heads):
                for n in range(t):
                    qn = q[bi, hi, n]  # (head_dim,)
                    s_hat = self.decision_s(qn)  # (1,)
                    o_hat = self.decision_o(qn)  # (1,)

                    # s in [0, max_window]
                    s = torch.sigmoid(s_hat) * max_window
                    # o in [-max_window, max_window]
                    o = torch.tanh(o_hat) * max_window

                    anchor = float(n) + float(o.item())
                    anchor_i = int(round(anchor))
                    s_i = int(round(float(s.item())))
                    L = max(0, anchor_i - s_i)
                    R = min(t - 1, anchor_i + s_i)
                    if L > R:
                        L = max(0, min(t - 1, anchor_i))
                        R = L

                    k_win = k[bi, hi, L : R + 1]  # (win, head_dim)
                    v_win = v[bi, hi, L : R + 1]  # (win, head_dim)
                    attn_logits = (k_win * qn.unsqueeze(0)).sum(dim=1) * scale  # (win,)
                    attn = F.softmax(attn_logits, dim=0)  # (win,)
                    out[bi, hi, n] = torch.matmul(attn.unsqueeze(0), v_win).squeeze(0)

        out = out.permute(0, 2, 1, 3).contiguous()  # (B, T, heads, head_dim)
        out = out.view(b, t, inner)  # (B, T, heads*head_dim)
        return out


class StageClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STA_TA1_Net(nn.Module):
    def __init__(
        self,
        n_mels: int,
        num_classes: int = 2,
        num_heads: int = 4,
        head_dim: int = 16,
    ):
        super().__init__()

        # 2D pooling aggregation:
        # Input x: (B, 1, F, T)
        # x_max: max_pool2d(k=3,s=2,p=1)
        # x_avg: avg_pool2d(k=3,s=2,p=1)
        # concat as channels => (B, 2, F', T')
        self.backbone = MobileNetV4LikeBackbone(in_ch=2, c1=32, c2=64, cp=96)

        # d_model derived from fp channels after frequency pooling.
        self.fp_channels = 96
        self.num_heads = num_heads
        self.head_dim = head_dim
        d_model = self.fp_channels
        if num_heads * head_dim != d_model:
            raise ValueError(
                f"Require num_heads*head_dim == fp_channels, got {num_heads}*{head_dim} != {d_model}. "
                "Adjust head_dim or fp channel size."
            )

        self.ta1 = TA1DynamicLocalAttention(d_model=d_model, num_heads=num_heads, head_dim=head_dim)

        # Stage classifiers:
        self.cls1 = StageClassifier(in_dim=32, hidden=64, dropout=0.2, num_classes=num_classes)
        self.cls2 = StageClassifier(in_dim=64, hidden=64, dropout=0.2, num_classes=num_classes)
        self.cls3 = StageClassifier(in_dim=num_heads * head_dim, hidden=128, dropout=0.2, num_classes=num_classes)

    def forward(self, x: torch.Tensor, max_window_tokens: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, 1, F, T)
        Returns:
          logits1, logits2, logits3, i_sp
        """
        # 1) pooling aggregation
        x_max = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_avg = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_agg = torch.cat([x_max, x_avg], dim=1)  # (B,2,F',T')

        # 2) backbone
        f1, f2, fp = self.backbone(x_agg)  # (B,32, F1, T1), (B,64,F2,T2), (B,96,Fp,Tp)

        # I_S1 / I_S2: global average pool over time+freq
        i_s1 = f1.mean(dim=(2, 3))  # (B,32)
        i_s2 = f2.mean(dim=(2, 3))  # (B,64)

        # I_Sp tokens: pool over frequency -> tokens over time
        # fp: (B, Cp=96, Fp, Tp) -> (B, Tp, Cp)
        tokens = fp.mean(dim=2).transpose(1, 2).contiguous()
        # TA1 outputs (B, Tp, heads*head_dim) == (B,Tp,96)
        h = self.ta1(tokens, max_window=max_window_tokens)
        i_sp = h.mean(dim=1)  # (B,96)

        logits1 = self.cls1(i_s1)
        logits2 = self.cls2(i_s2)
        logits3 = self.cls3(i_sp)
        return logits1, logits2, logits3, i_sp

    @torch.no_grad()
    def extract_embedding(
        self,
        x: torch.Tensor,
        max_window_tokens: int,
        feature_type: str = "concat_s1_s2",
    ) -> torch.Tensor:
        """
        Extract a fixed-length embedding for fusion without running classifiers.

        feature_type:
          - "i_s1": 32 dims
          - "i_s2": 64 dims
          - "concat_s1_s2": 96 dims (fast, skips TA1)
          - "i_sp": 96 dims (uses TA1)
        """
        # 1) pooling aggregation
        x_max = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_avg = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_agg = torch.cat([x_max, x_avg], dim=1)  # (B,2,F',T')

        # 2) backbone
        f1, f2, fp = self.backbone(x_agg)

        i_s1 = f1.mean(dim=(2, 3))  # (B,32)
        i_s2 = f2.mean(dim=(2, 3))  # (B,64)

        if feature_type == "i_s1":
            return i_s1
        if feature_type == "i_s2":
            return i_s2
        if feature_type == "concat_s1_s2":
            return torch.cat([i_s1, i_s2], dim=1)  # (B,96)
        if feature_type == "i_sp":
            tokens = fp.mean(dim=2).transpose(1, 2).contiguous()  # (B,Tp,96)
            h = self.ta1(tokens, max_window=max_window_tokens)  # (B,Tp,96)
            i_sp = h.mean(dim=1)  # (B,96)
            return i_sp

        raise ValueError(f"Unknown feature_type: {feature_type!r}")


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).sum().item()) / max(int(targets.numel()), 1)


def collate_batch(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    # xs: list[(n_mels,T)]
    x = torch.stack(xs, dim=0)  # (B,n_mels,T)
    x = x.unsqueeze(1)  # (B,1,n_mels,T)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def collate_extraction_batch(
    batch: Sequence[Tuple[torch.Tensor, int, str]],
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    xs, ys, keys = zip(*batch)
    x = torch.stack(xs, dim=0).unsqueeze(1)  # (B,1,n_mels,T)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y, list(keys)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    focal: FocalLoss,
    loss_weights: Tuple[float, float, float],
    max_window_tokens: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    total_correct = 0

    w1, w2, w3 = loss_weights
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits1, logits2, logits3, _ = model(x, max_window_tokens=max_window_tokens)
                l1 = focal(logits1, y)
                l2 = focal(logits2, y)
                l3 = focal(logits3, y)
                loss = w1 * l1 + w2 * l2 + w3 * l3
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits1, logits2, logits3, _ = model(x, max_window_tokens=max_window_tokens)
            l1 = focal(logits1, y)
            l2 = focal(logits2, y)
            l3 = focal(logits3, y)
            loss = w1 * l1 + w2 * l2 + w3 * l3
            loss.backward()
            optimizer.step()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs
        with torch.no_grad():
            pred = logits3.argmax(dim=1)
            total_correct += int((pred == y).sum().item())

    avg_loss = total_loss / max(total_n, 1)
    avg_acc = total_correct / max(total_n, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_window_tokens: int,
) -> Tuple[float, float]:
    model.eval()
    total_correct = 0
    total_n = 0
    total_loss = 0.0

    # We don't use focal loss here; only compute val acc & CE loss for readability.
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits1, logits2, logits3, _ = model(x, max_window_tokens=max_window_tokens)
        # Choose stage3 for accuracy
        loss = ce(logits3, y)
        total_loss += float(loss.item())

        pred = logits3.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total_n += int(y.size(0))

    acc = total_correct / max(total_n, 1)
    avg_loss = total_loss / max(total_n, 1)
    return acc, avg_loss


def list_subject_folders(audio_root: str) -> List[str]:
    subjects: List[str] = []
    for sd in sorted(os.listdir(audio_root)):
        sub = os.path.join(audio_root, sd)
        if os.path.isdir(sub) and is_subject_audio_folder(sd):
            subjects.append(sd)
    if not subjects:
        raise FileNotFoundError(f"No subject folders (0201/0202/0203) found under: {audio_root}")
    return subjects


@torch.no_grad()
def extract_sta_features_to_npz(
    audio_root: str,
    checkpoint_path: str,
    out_npz: str,
    segment_sec: float,
    stride_sec: float,
    sample_rate: int,
    n_mels: int,
    win_ms: float,
    hop_ms: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    feature_type: str = "concat_s1_s2",
    max_segments_per_file: Optional[int] = None,
) -> None:
    ck = torch.load(checkpoint_path, map_location="cpu")
    cfg = ck.get("config") or {}

    # Prefer checkpoint config to make sure feature dims match.
    n_mels_ck = int(cfg.get("n_mels", n_mels))
    num_heads = int(cfg.get("num_heads", 4))
    head_dim = int(cfg.get("head_dim", 16))
    num_classes = int(cfg.get("num_classes", 2)) if "num_classes" in cfg else 2

    model = STA_TA1_Net(
        n_mels=n_mels_ck,
        num_classes=num_classes,
        num_heads=num_heads,
        head_dim=head_dim,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    subject_list = list_subject_folders(audio_root)
    segments = build_extraction_segments_from_audio_root(
        audio_root=audio_root,
        subject_list=subject_list,
        segment_sec=segment_sec,
        stride_sec=stride_sec,
        max_segments_per_file=max_segments_per_file,
    )

    ds = SpeechSegmentsExtractionDataset(
        segments=segments,
        sample_rate=sample_rate,
        n_mels=n_mels_ck,
        win_ms=win_ms,
        hop_ms=hop_ms,
        segment_sec=segment_sec,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_extraction_batch,
        pin_memory=(device.type == "cuda"),
    )

    # Estimate token length from mel frames after pooling+backbone.
    dummy_mel = torch.zeros((1, 1, n_mels_ck, ds.expected_frames), device=device)
    x_max = F.max_pool2d(dummy_mel, kernel_size=3, stride=2, padding=1)
    x_avg = F.avg_pool2d(dummy_mel, kernel_size=3, stride=2, padding=1)
    x_agg = torch.cat([x_max, x_avg], dim=1)
    _, _, fp = model.backbone(x_agg)
    t_len = int(fp.shape[-1])
    max_window_tokens = max(1, int(t_len // 2))

    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}

    for x, _y, keys in dl:
        x = x.to(device)
        emb = model.extract_embedding(x, max_window_tokens=max_window_tokens, feature_type=feature_type)
        emb_np = emb.detach().cpu().numpy().astype(np.float32, copy=False)  # (B,D)
        for i, k in enumerate(keys):
            if k not in sums:
                sums[k] = emb_np[i].copy()
                counts[k] = 1
            else:
                sums[k] += emb_np[i]
                counts[k] += 1

    out_keys = sorted(sums.keys())
    sta = np.stack([sums[k] / float(counts[k]) for k in out_keys], axis=0).astype(np.float32, copy=False)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, keys=np.array(out_keys, dtype=object), sta=sta)
    print(f"[sta] saved keyed features: {out_npz} (N={sta.shape[0]}, D={sta.shape[1]})")


def main() -> None:
    p = argparse.ArgumentParser(description="STA_TA1 log-Mel + dynamic local attention training")
    p.add_argument(
        "--audio-root",
        type=str,
        default=_DEFAULT_AUDIO_ROOT,
        help=f"audio root，每人一个子目录（默认 52 人: {_DEFAULT_AUDIO_ROOT}）",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=0)

    # Feature extraction / training mode
    p.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "extract-features", "plot-log-mel"],
        help="train | extract-features | plot-log-mel（导出 control/depression 各一张 log-Mel 图）",
    )
    p.add_argument("--checkpoint", type=str, default=None, help="extract-features mode needs this checkpoint (.pth)")
    p.add_argument("--out-npz", type=str, default=None, help="output npz (keys + sta) for extract-features mode")
    p.add_argument(
        "--log-mel-png-hc",
        type=str,
        default=None,
        help="plot-log-mel: 保存 control 组首张可读 wav 的 log-Mel 图路径",
    )
    p.add_argument(
        "--log-mel-png-mdd",
        type=str,
        default=None,
        help="plot-log-mel: 保存 depression 组首张可读 wav 的 log-Mel 图路径",
    )
    p.add_argument(
        "--log-mel-plot-norm",
        action="store_true",
        help="plot-log-mel: 对 log-Mel 做与训练相同的 per-sample (mean/std) 归一化再画图",
    )
    p.add_argument(
        "--log-mel-cmap",
        type=str,
        default="magma",
        help="plot-log-mel: matplotlib colormap（默认 magma，与常见频谱图一致；可改 inferno / viridis 等）",
    )
    p.add_argument(
        "--feature-type",
        type=str,
        default="concat_s1_s2",
        choices=["concat_s1_s2", "i_s1", "i_s2", "i_sp"],
        help="Which STA embedding to export (i_sp uses TA1; concat_s1_s2 skips TA1 and is faster).",
    )

    # log-mel params
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=40)
    p.add_argument("--win-ms", type=float, default=25.0)
    p.add_argument("--hop-ms", type=float, default=10.0)

    # segment params
    p.add_argument("--segment-sec", type=float, default=3.0)
    p.add_argument("--stride-sec", type=float, default=3.0)
    p.add_argument("--max-segments-per-file", type=int, default=None)

    # TA1 params
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=24)  # must satisfy num_heads*head_dim == fp_channels(96)
    p.add_argument("--loss-weights", type=str, default="0.2,0.3,0.5", help="w1,w2,w3 for L1,L2,L3")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=None, help="alpha for positive class (binary). Example: 0.25")

    # training
    p.add_argument("--out-path", type=str, default=None, help="best checkpoint path")
    p.add_argument("--fp16", action="store_true", help="use AMP if CUDA available")

    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    if args.mode == "plot-log-mel":
        if not args.log_mel_png_hc and not args.log_mel_png_mdd:
            raise ValueError("plot-log-mel 需要至少指定 --log-mel-png-hc 或 --log-mel-png-mdd")
        try:
            sh, sm = try_save_log_mel_examples_by_depression_label(
                args.audio_root,
                out_hc=args.log_mel_png_hc,
                out_mdd=args.log_mel_png_mdd,
                sample_rate=args.sample_rate,
                n_mels=args.n_mels,
                win_ms=args.win_ms,
                hop_ms=args.hop_ms,
                segment_sec=args.segment_sec,
                apply_per_sample_norm=bool(args.log_mel_plot_norm),
                cmap=str(args.log_mel_cmap),
            )
        except ImportError as e:
            print(f"[plot-log-mel] skipped ({e})")
            raise SystemExit(1) from e
        if args.log_mel_png_hc:
            print(
                f"[plot-log-mel] saved control: {args.log_mel_png_hc}"
                if sh
                else f"[plot-log-mel] control figure not saved (no readable 0202/0203 wav?): {args.log_mel_png_hc!r}"
            )
        if args.log_mel_png_mdd:
            print(
                f"[plot-log-mel] saved depression: {args.log_mel_png_mdd}"
                if sm
                else f"[plot-log-mel] depression figure not saved (no readable 0201 wav?): {args.log_mel_png_mdd!r}"
            )
        return

    if args.mode == "extract-features":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in extract-features mode")
        out_npz = args.out_npz or os.path.join(os.getcwd(), "sta_ta1_keyed_features.npz")
        extract_sta_features_to_npz(
            audio_root=args.audio_root,
            checkpoint_path=args.checkpoint,
            out_npz=out_npz,
            segment_sec=args.segment_sec,
            stride_sec=args.stride_sec,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            win_ms=args.win_ms,
            hop_ms=args.hop_ms,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            feature_type=args.feature_type,
            max_segments_per_file=args.max_segments_per_file,
        )
        return

    out_dir = _DEFAULT_OUT_DIR
    if args.out_path:
        out_dir = os.path.dirname(os.path.abspath(args.out_path)) or _DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    loss_weights = tuple(float(x) for x in args.loss_weights.split(","))  # type: ignore[assignment]
    if len(loss_weights) != 3:
        raise ValueError("--loss-weights must be 3 numbers, like 0.2,0.3,0.5")

    # Split by subjects
    train_subjects, val_subjects = stratified_split_subjects(args.audio_root, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")

    # Build segments
    train_segments = build_segments_from_audio_root(
        args.audio_root,
        train_subjects,
        segment_sec=args.segment_sec,
        stride_sec=args.stride_sec,
        max_segments_per_file=args.max_segments_per_file,
    )
    val_segments = build_segments_from_audio_root(
        args.audio_root,
        val_subjects,
        segment_sec=args.segment_sec,
        stride_sec=args.stride_sec,
        max_segments_per_file=args.max_segments_per_file,
    )

    print(f"Train segments: {len(train_segments)} | Val segments: {len(val_segments)}")

    train_ds = SpeechSegmentsDataset(
        train_segments,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        segment_sec=args.segment_sec,
    )
    val_ds = SpeechSegmentsDataset(
        val_segments,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        segment_sec=args.segment_sec,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )

    model = STA_TA1_Net(
        n_mels=args.n_mels,
        num_classes=2,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    ).to(device)

    if args.fp16 and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    focal = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)

    # max_window_tokens depends on token length after pooling+backbone.
    # We estimate token length from mel frames by forward pass on a dummy input.
    with torch.no_grad():
        dummy_mel = torch.zeros((1, 1, args.n_mels, train_ds.expected_frames), device=device)
        # Do the same pooling as model: two 2x downsamples with stride2 (maxpool+avgpool)
        x_max = F.max_pool2d(dummy_mel, kernel_size=3, stride=2, padding=1)
        x_avg = F.avg_pool2d(dummy_mel, kernel_size=3, stride=2, padding=1)
        x_agg = torch.cat([x_max, x_avg], dim=1)
        _, _, fp = model.backbone(x_agg)
        # tokens = fp.mean(dim=2).transpose(1,2)
        t_len = fp.shape[-1]
    max_window_tokens = max(1, int(t_len // 2))
    print(f"Token length T'={t_len} | TA1 max_window={max_window_tokens}")

    best_acc = -1.0
    best_path = args.out_path or os.path.join(out_dir, "sta_ta1_best.pth")
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            focal=focal,
            loss_weights=loss_weights,
            max_window_tokens=max_window_tokens,
        )
        val_acc, val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            max_window_tokens=max_window_tokens,
        )
        print(
            f"[epoch {ep:03d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_val_acc": float(best_acc),
                    "config": vars(args),
                    "token_len": int(t_len),
                    "max_window_tokens": int(max_window_tokens),
                },
                best_path,
            )

    print("=" * 60)
    print(f"Best val acc: {best_acc * 100:.2f}%")
    print(f"Saved: {best_path}")

if __name__ == "__main__":
    main()

