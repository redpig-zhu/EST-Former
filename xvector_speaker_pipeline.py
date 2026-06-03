# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# =========================
# 默认输出目录：与本脚本同目录下的 xvector_ckpt（随文件移动自动变）
# =========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(_SCRIPT_DIR, "xvector_ckpt")

# 兰州 2015 语音：52 被试，目录结构与旧 multimodal_overlap/audio 一致
DEFAULT_AUDIO_ROOT = os.path.join(_SCRIPT_DIR, "audio_lanzhou_2015")
DEFAULT_SPEAKER_MANIFEST = os.path.join(_SCRIPT_DIR, "speaker_segments.csv")

SAMPLE_RATE = 16000
N_MEL = 40
FRAME_MS = 25.0
SHIFT_MS = 10.0
SEGMENT_SEC = 3.0
SLIDING_NORM_SEC = 3.0

TDNN_HIDDEN = 512
EMB_DIM = 128
TDNN_LAYERS = 5
TDNN_KERNEL = 5

AUDIO_EXTS = (".wav", ".wave", ".mp3", ".m4a", ".flac")


def depression_label_from_subject_folder(folder_name: str) -> int:
    """与 EEGRaw_data_numpyarray 一致：0201→MDD(1)，0202/0203→HC(0)。"""
    if folder_name.startswith("0201"):
        return 1
    if folder_name.startswith("0202") or folder_name.startswith("0203"):
        return 0
    raise ValueError(
        f"无法根据被试目录判断标签（需 0201→MDD 或 0202/0203→HC）: {folder_name!r}"
    )


def depression_label_from_wav_path(wav_path: str) -> int:
    parent = os.path.basename(os.path.dirname(os.path.normpath(wav_path)))
    return depression_label_from_subject_folder(parent)


def is_subject_audio_folder(folder_name: str) -> bool:
    """跳过 transcripts 等非 0201/0202/0203 被试目录。"""
    try:
        depression_label_from_subject_folder(folder_name)
        return True
    except ValueError:
        return False


def collect_wav_paths_from_audio_root(root: str) -> List[str]:
    """root 下每个子目录为一个被试，收集其下所有音频；路径按目录名、文件名排序。"""
    subdirs = sorted(
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and is_subject_audio_folder(d)
    )
    if not subdirs:
        raise FileNotFoundError(f"no subdirs under {root}")
    out: List[str] = []
    for sd in subdirs:
        folder = os.path.join(root, sd)
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(AUDIO_EXTS):
                continue
            out.append(os.path.normpath(os.path.join(folder, fn)))
    return out


# -------------------------
# Feature key alignment (for fusion)
# -------------------------

def wav_path_to_key(wav_path: str) -> str:
    """
    Align with your txt naming:
      pause/energy/tremor/{subject}_{wavstem}.txt
    """
    subject = os.path.basename(os.path.dirname(os.path.normpath(wav_path)))
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    return f"{subject}_{stem}"


# -------------------------
# Fbank
# -------------------------


def _load_wav_mono(path: str, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    try:
        import librosa

        y, sr = librosa.load(path, sr=None, mono=True)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return y.astype(np.float32), sr
    except Exception as e:
        raise RuntimeError(f"load_wav_mono failed for {path}: {e}") from e


def compute_fbank_kaldi(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    num_mel_bins: int = N_MEL,
) -> np.ndarray:
    """40 维 log Fbank，帧移 10ms、窗长 25ms（Kaldi 默认）。"""
    import torch

    try:
        import torchaudio.compliance.kaldi as kaldi
    except ImportError as e:
        raise ImportError("需要 torchaudio 以使用 kaldi.fbank") from e

    w = torch.from_numpy(waveform).float()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    feats = kaldi.fbank(
        w,
        sample_frequency=float(sample_rate),
        num_mel_bins=num_mel_bins,
        frame_length=FRAME_MS,
        frame_shift=SHIFT_MS,
    )
    return feats.numpy().astype(np.float32)


def compute_fbank_librosa_fallback(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    num_mel_bins: int = N_MEL,
) -> np.ndarray:
    """无 torchaudio 时的近似：log Mel，维度与帧率尽量对齐 Kaldi。"""
    import librosa

    n_fft = int(round(FRAME_MS * 1e-3 * sample_rate))
    hop = int(round(SHIFT_MS * 1e-3 * sample_rate))
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=num_mel_bins,
        fmin=20.0,
        fmax=sample_rate // 2,
        power=2.0,
    )
    log_mel = np.log(np.maximum(mel, 1e-10)).T.astype(np.float32)
    return log_mel


def compute_fbank(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    try:
        return compute_fbank_kaldi(waveform, sample_rate)
    except ImportError:
        return compute_fbank_librosa_fallback(waveform, sample_rate)


def _seconds_to_frames(sec: float) -> int:
    return max(1, int(round(sec * 1000.0 / SHIFT_MS)))


def sliding_mean_normalize(fbank: np.ndarray, window_sec: float = SLIDING_NORM_SEC) -> np.ndarray:
    """沿时间维做滑动均值归一化：每帧减去窗口内均值（窗口约 3 秒）。"""
    win = _seconds_to_frames(window_sec)
    x = np.asarray(fbank, dtype=np.float32)
    if x.size == 0:
        return x
    try:
        from scipy.ndimage import uniform_filter1d

        mu = uniform_filter1d(x, size=win, axis=0, mode="nearest")
        return x - mu
    except ImportError:
        pad = win // 2
        xpad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
        k = np.ones(win, dtype=np.float32) / float(win)
        out = np.empty_like(x, dtype=np.float32)
        for j in range(x.shape[1]):
            out[:, j] = x[:, j] - np.convolve(xpad[:, j], k, mode="valid")
        return out


def segment_waveform_fixed(
    y: np.ndarray,
    sample_rate: int,
    seg_sec: float = SEGMENT_SEC,
) -> List[np.ndarray]:
    """将波形切成约 seg_sec 秒一段；最后一段不足则零填充到 seg_sec。"""
    n = len(y)
    chunk = int(round(seg_sec * sample_rate))
    if chunk <= 0:
        return [y.copy()]
    segs: List[np.ndarray] = []
    for start in range(0, n, chunk):
        piece = y[start : start + chunk].astype(np.float32)
        if piece.size < chunk:
            pad = np.zeros(chunk - piece.size, dtype=np.float32)
            piece = np.concatenate([piece, pad], axis=0)
        segs.append(piece)
    return segs


def wav_to_fbank_normalized(
    path: str,
    seg_sec: float = SEGMENT_SEC,
    norm_window_sec: float = SLIDING_NORM_SEC,
) -> List[np.ndarray]:
    """单文件：重采样 -> 切约 3s -> 每段 Fbank -> 滑动均值归一化。返回多段 [T, 40]。"""
    y, sr = _load_wav_mono(path, SAMPLE_RATE)
    outs: List[np.ndarray] = []
    for seg in segment_waveform_fixed(y, sr, seg_sec):
        fb = compute_fbank(seg, sr)
        fb = sliding_mean_normalize(fb, norm_window_sec)
        outs.append(fb)
    return outs


def save_fbank_figure(
    fbank: np.ndarray,
    out_path: str,
    *,
    title: str = "Fbank (log-mel, sliding-mean normalized; 10ms hop, 25ms window)",
    wav_hint: str = "",
    dpi: int = 150,
) -> None:
    """将单段 Fbank [T, num_mel] 保存为 PNG 热图（与送入 TDNN 的特征一致，需 matplotlib）。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("保存 Fbank 示例图需要 matplotlib，请 pip install matplotlib") from e

    x = np.asarray(fbank, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        x.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="Greens",
    )
    ax.set_xlabel("Frame index (~10 ms/frame)")
    ax.set_ylabel("Mel bin (0..39)")
    ttl = title if not wav_hint else f"{title}\n{wav_hint}"
    ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="value")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def try_save_fbank_example_first_wav(audio_paths: Sequence[str], out_png: str) -> bool:
    """用列表中第一个成功读取的 wav 的第一段 3s 归一化 Fbank 保存示例图；失败返回 False。"""
    for p in audio_paths:
        if not os.path.isfile(p):
            continue
        try:
            segs = wav_to_fbank_normalized(p)
            if not segs or segs[0].size == 0:
                continue
            hint = os.path.basename(p)
            save_fbank_figure(segs[0], out_png, wav_hint=hint)
            return True
        except ImportError:
            raise
        except Exception:
            continue
    return False


def try_save_fbank_examples_by_depression_label(
    audio_paths: Sequence[str],
    *,
    out_hc: Optional[str] = None,
    out_mdd: Optional[str] = None,
) -> Tuple[bool, bool]:
    """
    按目录规则（0201→MDD，0202/0203→HC）各取扫盘顺序下首个可读 wav 的第一段 3s 归一化 Fbank。
    返回 (是否保存 HC 图, 是否保存 MDD 图)。
    """
    if not out_hc and not out_mdd:
        return False, False
    saved_hc = False
    saved_mdd = False
    for p in audio_paths:
        if not os.path.isfile(p):
            continue
        try:
            label = depression_label_from_wav_path(p)
        except ValueError:
            continue
        want = (label == 0 and out_hc and not saved_hc) or (label == 1 and out_mdd and not saved_mdd)
        if not want:
            continue
        try:
            segs = wav_to_fbank_normalized(p)
            if not segs or segs[0].size == 0:
                continue
            if label == 0 and out_hc and not saved_hc:
                save_fbank_figure(segs[0], out_hc, title="(a) Control")
                saved_hc = True
            elif label == 1 and out_mdd and not saved_mdd:
                save_fbank_figure(segs[0], out_mdd, title="(b) Depression")
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


def _run_fbank_example_exports(
    audio_paths: Sequence[str],
    *,
    fbank_example_png: Optional[str] = None,
    fbank_example_png_hc: Optional[str] = None,
    fbank_example_png_mdd: Optional[str] = None,
) -> None:
    """extract 前导出 Fbank 示意图：优先按 HC/MDD 各一张；否则仅 --fbank-example-png 首张 wav。"""
    if fbank_example_png_hc or fbank_example_png_mdd:
        try:
            sh, sm = try_save_fbank_examples_by_depression_label(
                audio_paths, out_hc=fbank_example_png_hc, out_mdd=fbank_example_png_mdd
            )
            if fbank_example_png_hc:
                if sh:
                    print(f"[extract-xvector] saved fbank example (HC): {fbank_example_png_hc}")
                else:
                    print(
                        f"[extract-xvector] no HC fbank figure saved (no readable 0202/0203 wav?): "
                        f"{fbank_example_png_hc!r}"
                    )
            if fbank_example_png_mdd:
                if sm:
                    print(f"[extract-xvector] saved fbank example (MDD): {fbank_example_png_mdd}")
                else:
                    print(
                        f"[extract-xvector] no MDD fbank figure saved (no readable 0201 wav?): "
                        f"{fbank_example_png_mdd!r}"
                    )
        except ImportError as e:
            print(f"[extract-xvector] skip fbank figure ({e})")
    elif fbank_example_png:
        try:
            if try_save_fbank_example_first_wav(audio_paths, fbank_example_png):
                print(f"[extract-xvector] saved fbank example figure: {fbank_example_png}")
            else:
                print(
                    f"[extract-xvector] could not save fbank example to {fbank_example_png} "
                    "(no readable wav)"
                )
        except ImportError as e:
            print(f"[extract-xvector] skip fbank figure ({e})")


# -------------------------
# TDNN + stats pooling + emb6/emb7
# -------------------------


class StatisticsPooling(nn.Module):
    """沿时间维 mean + std（可带 mask）。"""

    @staticmethod
    def forward_masked(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, c, t = x.shape
        device = x.device
        rng = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        mask = (rng < lengths.unsqueeze(1)).float().unsqueeze(1)
        denom = mask.sum(dim=2).clamp(min=1e-8)
        mean = (x * mask).sum(dim=2) / denom
        xc = x - mean.unsqueeze(2)
        var = ((xc * mask) ** 2).sum(dim=2) / denom
        std = torch.sqrt(var + 1e-8)
        return torch.cat([mean, std], dim=1)


class TDNNXVectorSpeakerNet(nn.Module):
    """
    TDNN(5 层) -> stats pooling -> Linear(2H->128) embedding6 -> Linear(128->128) embedding7 -> speaker logits
    推理/抑郁特征：使用 embedding6 作为 128 维 x-vector（与 Step C/D 一致）。
    """

    def __init__(
        self,
        in_dim: int = N_MEL,
        hidden: int = TDNN_HIDDEN,
        emb_dim: int = EMB_DIM,
        num_speakers: int = 100,
        num_tdnn_layers: int = TDNN_LAYERS,
        kernel: int = TDNN_KERNEL,
    ):
        super().__init__()
        pad = kernel // 2
        blocks: List[nn.Module] = []
        c_in = in_dim
        for _ in range(num_tdnn_layers):
            blocks.extend(
                [
                    nn.Conv1d(c_in, hidden, kernel_size=kernel, padding=pad),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(hidden),
                ]
            )
            c_in = hidden
        self.tdnn = nn.Sequential(*blocks)
        stats_dim = hidden * 2
        self.embedding6 = nn.Linear(stats_dim, emb_dim)
        self.bn6 = nn.BatchNorm1d(emb_dim)
        self.embedding7 = nn.Linear(emb_dim, emb_dim)
        self.bn7 = nn.BatchNorm1d(emb_dim)
        self.speaker_head = nn.Linear(emb_dim, num_speakers)

    def forward_tdnn(self, x_bt_f: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, C, T)
        x = x_bt_f.transpose(1, 2)
        h = self.tdnn(x)
        return StatisticsPooling.forward_masked(h, lengths)

    def forward(
        self,
        x_bt_f: torch.Tensor,
        lengths: torch.Tensor,
        return_logits: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        stats = self.forward_tdnn(x_bt_f, lengths)
        e6 = self.bn6(self.embedding6(stats))
        e6 = F.relu(e6)
        e7 = self.bn7(self.embedding7(e6))
        e7 = F.relu(e7)
        if not return_logits:
            return e6
        logits = self.speaker_head(e7)
        return e6, logits

    def forward_xvector(self, x_bt_f: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            stats = self.forward_tdnn(x_bt_f, lengths)
            e6 = self.bn6(self.embedding6(stats))
            e6 = F.relu(e6)
        return e6


# -------------------------
# Dataset & collate
# -------------------------


@dataclass
class SegmentRow:
    path: str
    speaker_id: int
    start_sec: float = 0.0
    end_sec: Optional[float] = None


class SpeakerSegmentDataset(Dataset):
    """manifest CSV: path,speaker_id[,start_sec,end_sec]"""

    def __init__(self, rows: Sequence[SegmentRow]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        r = self.rows[idx]
        y, sr = _load_wav_mono(r.path, SAMPLE_RATE)
        if r.end_sec is not None:
            a = int(max(0.0, r.start_sec) * sr)
            b = int(min(len(y), r.end_sec * sr))
            y = y[a:b]
        elif r.start_sec > 0:
            a = int(r.start_sec * sr)
            y = y[a:]
        # 单条样本对应一段波形：不足 3s 则零填充到 3s 再提 Fbank（与 prepare-manifest 一致）
        segs = segment_waveform_fixed(y, sr, SEGMENT_SEC)
        seg = segs[0] if segs else np.zeros(int(SEGMENT_SEC * sr), dtype=np.float32)
        fb = compute_fbank(seg, sr)
        fb = sliding_mean_normalize(fb)
        t = torch.from_numpy(fb)
        return t, int(r.speaker_id), t.shape[0]


def collate_speaker_batch(
    batch: Sequence[Tuple[torch.Tensor, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats, labels, lens = zip(*batch)
    lengths = torch.tensor(lens, dtype=torch.long)
    padded = pad_sequence(feats, batch_first=True)
    y = torch.tensor(labels, dtype=torch.long)
    return padded, y, lengths


def load_manifest_csv(path: str) -> List[SegmentRow]:
    rows: List[SegmentRow] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            p = raw.get("path") or raw.get("wav_path") or raw.get("audio_path")
            if not p or not os.path.isfile(p):
                continue
            spk = int(raw["speaker_id"])
            st = float(raw.get("start_sec") or 0.0)
            en = raw.get("end_sec")
            en_f = float(en) if en not in (None, "") else None
            rows.append(SegmentRow(path=p.strip(), speaker_id=spk, start_sec=st, end_sec=en_f))
    return rows


def build_speaker_label_map(rows: Sequence[SegmentRow]) -> Tuple[Dict[int, int], int]:
    uniq = sorted({r.speaker_id for r in rows})
    m = {sid: i for i, sid in enumerate(uniq)}
    return m, len(uniq)


# -------------------------
# 准备 manifest：按目录名作为 speaker id（整数需自行映射）
# -------------------------


def prepare_manifest_from_dirs(
    root: str,
    out_csv: str,
    speaker_id_map: Optional[Dict[str, int]] = None,
) -> None:
    """
    root 下每个子目录名为 speaker（若 speaker_id_map 未给，则按字典序编号 0..K-1）。
    每个 wav 切 3s 段，一行一段。
    """
    subdirs = sorted(
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and is_subject_audio_folder(d)
    )
    if not subdirs:
        raise FileNotFoundError(f"no subject subdirs (0201/0202/0203) under {root}")

    if speaker_id_map is None:
        speaker_id_map = {name: i for i, name in enumerate(subdirs)}

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    skipped_bad_audio = 0
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "speaker_id", "start_sec", "end_sec"])
        for sd in subdirs:
            sid = speaker_id_map.get(sd)
            if sid is None:
                continue
            folder = os.path.join(root, sd)
            for fn in sorted(os.listdir(folder)):
                if not fn.lower().endswith(AUDIO_EXTS):
                    continue
                path = os.path.join(folder, fn)
                try:
                    y, sr = _load_wav_mono(path, SAMPLE_RATE)
                except Exception as e:
                    skipped_bad_audio += 1
                    print(f"[prepare-manifest] skip unreadable audio: {path} ({e})")
                    continue
                dur = len(y) / float(sr)
                start = 0.0
                while start + 1e-3 < dur:
                    end = min(start + SEGMENT_SEC, dur)
                    w.writerow([path, sid, f"{start:.4f}", f"{end:.4f}"])
                    start += SEGMENT_SEC
    if skipped_bad_audio > 0:
        print(f"[prepare-manifest] skipped unreadable files: {skipped_bad_audio}")


# -------------------------
# 训练 speaker / 保存
# -------------------------


def train_speaker_classifier(
    manifest_csv: str,
    out_dir: str,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 42,
) -> str:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows = load_manifest_csv(manifest_csv)
    if not rows:
        raise RuntimeError("manifest 为空或路径无效")

    spk_map, _ = build_speaker_label_map(rows)
    rows_mapped = [
        SegmentRow(
            path=r.path,
            speaker_id=spk_map[r.speaker_id],
            start_sec=r.start_sec,
            end_sec=r.end_sec,
        )
        for r in rows
    ]
    num_spk = len(spk_map)
    ds = SpeakerSegmentDataset(rows_mapped)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_speaker_batch,
        num_workers=0,
    )

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TDNNXVectorSpeakerNet(num_speakers=num_spk).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    os.makedirs(out_dir, exist_ok=True)
    best_loss = float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    losses: List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for x, y, lens in dl:
            x = x.to(dev)
            y = y.to(dev)
            lens = lens.to(dev)
            opt.zero_grad()
            _, logits = model(x, lens, return_logits=True)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()) * x.size(0)
            n += x.size(0)
        avg = tot / max(n, 1)
        losses.append(float(avg))
        print(f"[speaker] epoch {ep}/{epochs} loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_speakers": num_spk,
                    "spk_map": spk_map,
                    "config": {
                        "n_mel": N_MEL,
                        "tdnn_hidden": TDNN_HIDDEN,
                        "emb_dim": EMB_DIM,
                        "tdnn_layers": TDNN_LAYERS,
                    },
                },
                best_path,
            )

    return best_path


# -------------------------
# 抽取 x-vector（embedding6）
# -------------------------


def load_speaker_net(checkpoint_path: str, device: Optional[str] = None) -> TDNNXVectorSpeakerNet:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint_path, map_location=dev)
    cfg = ck.get("config") or {}
    num_spk = int(ck.get("num_speakers", 2))
    model = TDNNXVectorSpeakerNet(
        num_speakers=num_spk,
        hidden=int(cfg.get("tdnn_hidden", TDNN_HIDDEN)),
        emb_dim=int(cfg.get("emb_dim", EMB_DIM)),
        num_tdnn_layers=int(cfg.get("tdnn_layers", TDNN_LAYERS)),
    )
    model.load_state_dict(ck["model"])
    model.to(dev)
    model.eval()
    return model


def fbank_tensors_from_wav(path: str, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """每条 wav 可能多段 3s；返回 [(padded_fbank_B1T40, lengths), ...]。"""
    segs = wav_to_fbank_normalized(path)
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if not segs:
        z = torch.zeros(1, 1, N_MEL, device=device)
        out.append((z, torch.tensor([1], dtype=torch.long, device=device)))
        return out
    for fb in segs:
        t = torch.from_numpy(fb).unsqueeze(0).to(device)
        lens = torch.tensor([t.size(1)], dtype=torch.long, device=device)
        out.append((t, lens))
    return out


def extract_xvector_file(model: TDNNXVectorSpeakerNet, wav_path: str, device: torch.device) -> np.ndarray:
    """长音频：对每段 3s Fbank 分别算 embedding6，再对段取均值得到一条 128 维向量。"""
    chunks: List[np.ndarray] = []
    for x, lens in fbank_tensors_from_wav(wav_path, device):
        vec = model.forward_xvector(x, lens)
        chunks.append(vec.squeeze(0).cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros(EMB_DIM, dtype=np.float32)
    return np.mean(np.stack(chunks, axis=0), axis=0)


def extract_xvector_list(
    checkpoint_path: str,
    out_npy: str,
    audio_root: str,
    device: Optional[str] = None,
    fbank_example_png: Optional[str] = None,
    fbank_example_png_hc: Optional[str] = None,
    fbank_example_png_mdd: Optional[str] = None,
) -> np.ndarray:
    """audio_root：被试每人一个子目录，其下 wav（与 prepare-manifest 同一套目录）。"""
    paths = collect_wav_paths_from_audio_root(audio_root)

    _run_fbank_example_exports(
        paths,
        fbank_example_png=fbank_example_png,
        fbank_example_png_hc=fbank_example_png_hc,
        fbank_example_png_mdd=fbank_example_png_mdd,
    )

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_speaker_net(checkpoint_path, str(dev))
    feats: List[np.ndarray] = []
    skipped_bad_audio = 0
    for p in paths:
        if not os.path.isfile(p):
            feats.append(np.zeros(EMB_DIM, dtype=np.float32))
            continue
        try:
            feats.append(extract_xvector_file(model, p, dev))
        except Exception as e:
            skipped_bad_audio += 1
            print(f"[extract-xvector] skip unreadable audio and fill zeros: {p} ({e})")
            feats.append(np.zeros(EMB_DIM, dtype=np.float32))
    X = np.stack(feats, axis=0)
    os.makedirs(os.path.dirname(out_npy) or ".", exist_ok=True)
    np.save(out_npy, X)
    if skipped_bad_audio > 0:
        print(f"[extract-xvector] unreadable files filled with zero vectors: {skipped_bad_audio}")
    return X


# -------------------------
# Export keyed xvector features (for fusion)
# -------------------------

def extract_xvector_keyed_npz(
    checkpoint_path: str,
    out_npz: str,
    audio_root: str,
    device: Optional[str] = None,
    fbank_example_png: Optional[str] = None,
    fbank_example_png_hc: Optional[str] = None,
    fbank_example_png_mdd: Optional[str] = None,
) -> str:
    """
    Export a keyed npz for fusion:
      keys: list[str] where each key = <subject>_<wavstem>
      xvector: (N,128)
    """
    paths = collect_wav_paths_from_audio_root(audio_root)
    _run_fbank_example_exports(
        paths,
        fbank_example_png=fbank_example_png,
        fbank_example_png_hc=fbank_example_png_hc,
        fbank_example_png_mdd=fbank_example_png_mdd,
    )
    keys: List[str] = [wav_path_to_key(p) for p in paths]

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_speaker_net(checkpoint_path, str(dev))

    feats: List[np.ndarray] = []
    skipped_bad_audio = 0
    for p in paths:
        if not os.path.isfile(p):
            feats.append(np.zeros(EMB_DIM, dtype=np.float32))
            continue
        try:
            feats.append(extract_xvector_file(model, p, dev))
        except Exception as e:
            skipped_bad_audio += 1
            print(f"[extract-xvector] skip unreadable audio and fill zeros: {p} ({e})")
            feats.append(np.zeros(EMB_DIM, dtype=np.float32))

    X = np.stack(feats, axis=0).astype(np.float32, copy=False)  # (N,128)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, keys=np.array(keys, dtype=object), xvector=X)
    if skipped_bad_audio > 0:
        print(f"[extract-xvector] unreadable files filled with zero vectors: {skipped_bad_audio}")
    print(f"[extract-xvector] saved keyed features: {out_npz}")
    return out_npz


# -------------------------
# Step E: 抑郁 MLP / 线性分类
# -------------------------


class DepressionHead(nn.Module):
    def __init__(self, in_dim: int = EMB_DIM, hidden: int = 64, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_depression_head(
    feat_npy: str,
    audio_root: str,
    out_path: str,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 42,
) -> float:
    """
    与 extract-xvector 使用同一 audio_root、同一扫盘顺序；标签由子目录名在内存中判定
    （0201→MDD=1，0202/0203→HC=0），不写 CSV。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = np.load(feat_npy)
    paths = collect_wav_paths_from_audio_root(audio_root)
    y = np.array([depression_label_from_wav_path(p) for p in paths], dtype=np.int64)
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"标签数 {y.shape[0]}（来自 {audio_root} 扫盘）与特征行数 {X.shape[0]} 不一致；"
            "请保证 ③④ 使用同一 --root 且中间未改音频目录"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    n = len(ds)
    n_tr = int(n * 0.85)
    tr, va = torch.utils.data.random_split(ds, [n_tr, n - n_tr], generator=torch.Generator().manual_seed(seed))
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)

    model = DepressionHead(in_dim=X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    train_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        tot_n = 0
        tr_correct = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * xb.size(0)
            tot_n += xb.size(0)
            tr_pred = logits.argmax(dim=1)
            tr_correct += int((tr_pred == yb).sum().item())
        train_loss = tot_loss / max(tot_n, 1)
        train_acc = tr_correct / max(tot_n, 1)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += xb.size(0)
        acc = correct / max(total, 1)
        train_losses.append(float(train_loss))
        train_accs.append(float(train_acc))
        val_accs.append(float(acc))
        print(
            f"[depression] epoch {ep}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={acc:.4f}"
        )
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)

    print("")
    print("=" * 60)
    print(f"二分类验证集准确率（best，用于选权重）: {best_acc * 100:.2f}%  （小数: {best_acc:.4f}）")
    print(f"已保存分类头: {out_path}")
    print("=" * 60)
    return float(best_acc)


# -------------------------
# CLI
# -------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Fbank + TDNN x-vector 声纹与抑郁分类流程")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prepare-manifest", help="从按说话人分目录的数据生成 manifest CSV（默认 root=audio_lanzhou_2015）")
    p1.add_argument("--root", type=str, default=DEFAULT_AUDIO_ROOT, help=f"说话人根目录，子目录名作为 speaker id（默认: {DEFAULT_AUDIO_ROOT}）")
    p1.add_argument("--out", type=str, default=DEFAULT_SPEAKER_MANIFEST, help="输出的 speaker_segments.csv 路径")

    p2 = sub.add_parser("pretrain-speaker", help="训练 speaker TDNN")
    p2.add_argument("--manifest", type=str, default=DEFAULT_SPEAKER_MANIFEST, help="① 生成的 CSV，默认本目录 speaker_segments.csv")
    p2.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p2.add_argument("--epochs", type=int, default=40)
    p2.add_argument("--batch-size", type=int, default=16)
    p2.add_argument("--lr", type=float, default=1e-3)
    p3 = sub.add_parser("extract-xvector", help="从按被试分目录的 audio 根目录抽 128 维 x-vector")
    p3.add_argument("--checkpoint", type=str, required=True)
    p3.add_argument("--root", type=str, default=DEFAULT_AUDIO_ROOT, help=f"被试分目录的根（默认与 prepare-manifest 相同: {DEFAULT_AUDIO_ROOT}）")
    p3.add_argument("--out-npy", type=str, required=True)
    p3.add_argument(
        "--fbank-example-png",
        type=str,
        default=None,
        help="可选：保存首张可读 wav 的归一化 Fbank（与下方 HC/MDD 二选一；若同时指定 HC/MDD 则忽略本项）",
    )
    p3.add_argument(
        "--fbank-example-png-hc",
        type=str,
        default=None,
        help="可选：健康对照组首张可读 wav（0202/0203）Fbank 热图路径",
    )
    p3.add_argument(
        "--fbank-example-png-mdd",
        type=str,
        default=None,
        help="可选：抑郁组首张可读 wav（0201）Fbank 热图路径",
    )

    p3b = sub.add_parser("extract-xvector-features", help="导出带 keys 的 xvector 特征 npz（用于融合）")
    p3b.add_argument("--checkpoint", type=str, required=True)
    p3b.add_argument("--root", type=str, default=DEFAULT_AUDIO_ROOT, help=f"被试分目录的根（默认与 prepare-manifest 相同: {DEFAULT_AUDIO_ROOT}）")
    p3b.add_argument("--out-npz", type=str, required=True, help="输出 npz：包含 keys + xvector")
    p3b.add_argument(
        "--fbank-example-png",
        type=str,
        default=None,
        help="可选：保存首张可读 wav 的归一化 Fbank（与 HC/MDD 二选一；若同时指定 HC/MDD 则忽略本项）",
    )
    p3b.add_argument(
        "--fbank-example-png-hc",
        type=str,
        default=None,
        help="可选：健康对照组首张可读 wav（0202/0203）Fbank 热图路径",
    )
    p3b.add_argument(
        "--fbank-example-png-mdd",
        type=str,
        default=None,
        help="可选：抑郁组首张可读 wav（0201）Fbank 热图路径",
    )

    p4 = sub.add_parser("train-depression", help="128 维特征 + MLP 抑郁分类")
    p4.add_argument("--feat-npy", type=str, required=True)
    p4.add_argument("--root", type=str, default=DEFAULT_AUDIO_ROOT, help=f"须与 extract-xvector 相同（默认: {DEFAULT_AUDIO_ROOT}）")
    p4.add_argument("--out", type=str, default=os.path.join(DEFAULT_OUT_DIR, "depression_mlp.pt"))
    p4.add_argument("--epochs", type=int, default=80)
    p5 = sub.add_parser("run-full", help="一条龙：跑完抽 x-vector + 训练抑郁头，并在最后输出准确率")
    p5.add_argument("--audio-root", type=str, default=DEFAULT_AUDIO_ROOT, help=f"被试音频根目录（默认: {DEFAULT_AUDIO_ROOT}）")
    p5.add_argument("--ckpt-dir", type=str, default=DEFAULT_OUT_DIR, help=f"声纹与分类输出目录（默认: {DEFAULT_OUT_DIR}）")
    p5.add_argument(
        "--xvector-ckpt",
        type=str,
        default=None,
        help="若提供已有 best.pt 路径，则跳过预训练；否则默认使用 --ckpt-dir/best.pt",
    )
    p5.add_argument(
        "--manifest-out",
        type=str,
        default=DEFAULT_SPEAKER_MANIFEST,
        help="仅在需要预训练时生成 speaker_segments.csv",
    )
    p5.add_argument("--xvec-out", type=str, default=os.path.join(DEFAULT_OUT_DIR, "xvec128.npy"), help="抽出的 x-vector 特征文件")
    p5.add_argument("--depression-out", type=str, default=os.path.join(DEFAULT_OUT_DIR, "depression_mlp.pt"), help="训练出的抑郁分类器权重")
    p5.add_argument("--xvector-epochs", type=int, default=40)
    p5.add_argument("--xvector-batch-size", type=int, default=16)
    p5.add_argument("--xvector-lr", type=float, default=1e-3)
    p5.add_argument("--depression-epochs", type=int, default=80)
    p5.add_argument("--seed", type=int, default=42)
    p5.add_argument("--force-pretrain", action="store_true", help="即使 best.pt 已存在也强制重新预训练")
    p5.add_argument(
        "--fbank-example-png",
        type=str,
        default=None,
        help="可选：首张可读 wav Fbank；若指定 --fbank-example-png-hc / -mdd 则忽略本项",
    )
    p5.add_argument(
        "--fbank-example-png-hc",
        type=str,
        default=None,
        help="可选：健康对照（0202/0203）首张可读 wav 的 Fbank 热图",
    )
    p5.add_argument(
        "--fbank-example-png-mdd",
        type=str,
        default=None,
        help="可选：抑郁（0201）首张可读 wav 的 Fbank 热图",
    )

    args = p.parse_args()
    if args.cmd == "prepare-manifest":
        prepare_manifest_from_dirs(args.root, args.out)
        print(f"wrote {args.out}")
    elif args.cmd == "pretrain-speaker":
        path = train_speaker_classifier(
            args.manifest,
            args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        print(f"done best={path}")
    elif args.cmd == "extract-xvector":
        extract_xvector_list(
            args.checkpoint,
            args.out_npy,
            audio_root=args.root,
            fbank_example_png=args.fbank_example_png,
            fbank_example_png_hc=args.fbank_example_png_hc,
            fbank_example_png_mdd=args.fbank_example_png_mdd,
        )
        print(f"saved {args.out_npy}")
    elif args.cmd == "extract-xvector-features":
        extract_xvector_keyed_npz(
            checkpoint_path=args.checkpoint,
            out_npz=args.out_npz,
            audio_root=args.root,
            fbank_example_png=args.fbank_example_png,
            fbank_example_png_hc=args.fbank_example_png_hc,
            fbank_example_png_mdd=args.fbank_example_png_mdd,
        )
    elif args.cmd == "train-depression":
        train_depression_head(
            args.feat_npy,
            args.root,
            args.out,
            epochs=args.epochs,
        )
    elif args.cmd == "run-full":
        audio_root = os.path.normpath(args.audio_root)
        if not os.path.isdir(audio_root):
            raise FileNotFoundError(
                f"音频目录不存在: {audio_root!r}\n"
                "示例里的 ... 是省略写法，请换成你机器上的真实路径；"
                "若数据就在本仓库默认位置，可直接去掉 --audio-root 使用脚本默认值。"
            )
        ckpt_path = args.xvector_ckpt or os.path.join(args.ckpt_dir, "best.pt")
        if args.force_pretrain or (not os.path.isfile(ckpt_path)):
            # 只有在缺少 best.pt 时才生成 manifest 并预训练
            prepare_manifest_from_dirs(audio_root, args.manifest_out)
            ckpt_path = train_speaker_classifier(
                args.manifest_out,
                args.ckpt_dir,
                epochs=args.xvector_epochs,
                batch_size=args.xvector_batch_size,
                lr=args.xvector_lr,
                seed=args.seed,
            )

        extract_xvector_list(
            ckpt_path,
            args.xvec_out,
            audio_root=audio_root,
            fbank_example_png=args.fbank_example_png,
            fbank_example_png_hc=args.fbank_example_png_hc,
            fbank_example_png_mdd=args.fbank_example_png_mdd,
        )
        # 最终会在 train_depression_head 里打印“二分类验证集准确率”
        train_depression_head(
            args.xvec_out,
            audio_root,
            args.depression_out,
            epochs=args.depression_epochs,
            seed=args.seed,
        )
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
