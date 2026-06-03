

import os
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Set

import whisper

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tokenizer import Bert768Encoder

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# =========================
# 路径设置
# - 音频输入：.../Data_preprocessed_EEGSpeech/audio_lanzhou_2015（被试子目录 + wav）
# - ASR 输出：.../multimodal_overlap/save_audio_ASR；SKIP_IF_EXISTS=True 时不覆盖已有 txt
# =========================
audio_root = r"D:\pigfile\redpig_code\eegaudio\Data_preprocessed_EEGSpeech\audio_lanzhou_2015"
save_base_path = r"D:\pigfile\redpig_code\eegaudio\multimodal_overlap\save_audio_ASR"

save_path_mdd = os.path.join(save_base_path, "MDD")
save_path_control = os.path.join(save_base_path, "Control")
os.makedirs(save_path_mdd, exist_ok=True)
os.makedirs(save_path_control, exist_ok=True)

# =========================
# ASR 参数
# =========================
WHISPER_MODEL = "medium"   # base/small/medium/large
LANGUAGE = "zh"           # "zh"/"en"/None
TASK = "transcribe"       # transcribe / translate
AUDIO_EXTS = (".wav", ".wave", ".mp3", ".m4a")
VERBOSE_ASR = False
PRINT_ASR_FILE_LOG = False  # False: 不打印每条音频转写日志，只保留汇总
# 若为 True：检测到已存在 ASR 文本时跳过单文件转写
SKIP_IF_EXISTS = True
# 若为 True：忽略已有 txt，强制重跑全部 ASR（用于修复转写质量）
FORCE_RETRANSCRIBE = False
# Whisper 解码参数（更稳的转写设置）
ASR_FP16 = torch.cuda.is_available()
ASR_BEAM_SIZE = 5
ASR_BEST_OF = 5
ASR_TEMPERATURE = 0.0
# 精简输出：True 时不导出 NPY，仅打印训练准确率并可选保存分类头
MINIMAL_OUTPUT = True

RUN_CLASSIFIER_TRAIN = True
# 最小输出模式下是否保存分类头权重
SAVE_CLASSIFIER_HEAD = False
# =========================
# BERT 768 参数（只做特征提取）
# =========================
BERT_NAME = "bert-base-chinese"  # hidden=768
MAX_LEN = 256
FEAT_BATCH = 32
L2_NORMALIZE = True  # 是否对 768 特征做 L2 normalize
REUSE_CACHED_FEATURES = True  # True: 若存在 bert768_features.npy 则直接复用，不重新提特征


# =========================
EXPORT_BERT_FEATS_NPZ = True
BERT_FEATS_NPZ_PATH = os.path.join(save_base_path, "bert_feats_keyed.npz")
# =========================
# 端到端微调参数（大方向：直接微调预训练模型）
# =========================
TRAIN_MODE = "feature"  # "feature" / "finetune"
FINETUNE_MODEL_CANDIDATES = [
    r"D:\models\bert-base-chinese",
    "bert-base-chinese",
]
FT_MAX_LEN = 256
FT_BATCH = 16
FT_EPOCHS = 12
FT_LR = 2e-5
FT_WEIGHT_DECAY = 1e-2
# 验证集小、波动大：早停耐心略大，避免一次掉点就把好权重丢掉
FT_PATIENCE = 8
# "none"=固定 lr 跑满（与你曾跑到 ~0.77 的设置一致）；
# "plateau"=按 val_acc 降 lr，小验证集易被单次波动误触发，默认不用
FT_SCHEDULER_MODE = "none"  # "plateau" / "none"
FT_PLATEAU_FACTOR = 0.5
FT_PLATEAU_PATIENCE = 2
FT_MIN_LR = 1e-7
FT_GRAD_CLIP = 1.0
FT_FREEZE_EPOCHS = 0  # 0=首轮即全量微调（与你之前 0.76+ 的设置一致）
# 0=关闭；0.05~0.1 有时能缓解过拟合、平滑小验证集波动（可试）
FT_LABEL_SMOOTHING = 0.0
# 多次换随机种子划分，看最好/平均 val（单次 0.72 vs 0.77 可能只是划分运气）
FT_MULTI_SEED_EVAL = False
FT_EVAL_SEEDS = [42, 43, 44, 45, 46]

# =========================
# 分类器训练参数（单模型）
# =========================
USE_CLASS_WEIGHT = True
EPOCHS = 80
CLS_BATCH = 32
LR = 8e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 0  # <=0 表示关闭早停，固定跑满 EPOCHS
SEED = 42
CLASSIFIER_TYPE = "auto"   # "linear" / "mlp" / "auto"(自动选验证集更高者)
HIDDEN_DIM = 384
DROPOUT = 0.30
LR_LINEAR = 2e-4
LR_MLP = 6e-4
LINEAR_LR_CANDIDATES = [1e-4, 2e-4, 3e-4]
MLP_LR_CANDIDATES = [3e-4, 6e-4, 8e-4]
WD_CANDIDATES = [1e-5, 1e-4, 3e-4]


# =========================
# 工具函数
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_subject_id(subject_id: str) -> str:
    subject_id = str(subject_id).strip()
    if len(subject_id) < 8:
        subject_id = "0" * (8 - len(subject_id)) + subject_id
    return subject_id


def list_subject_ids_from_audio_root(root: str, id_prefix: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and str(name).startswith(id_prefix):
            out.append(format_subject_id(name))
    return out


def find_subject_folder(base_audio_path: str, subject_id_8: str) -> Optional[str]:
    candidates = [
        os.path.join(base_audio_path, subject_id_8),
        os.path.join(base_audio_path, subject_id_8.lstrip("0")),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isdir(p):
            return p
    return None

def list_audio_files(folder: str) -> List[str]:
    return [fn for fn in os.listdir(folder) if fn.lower().endswith(AUDIO_EXTS)]


def scan_missing_asr_txt(subject_ids: List[str], save_path: str) -> Tuple[int, Set[str]]:

    missing_n = 0
    subjects_hit: Set[str] = set()
    for subject_id in subject_ids:
        fpath = find_subject_folder(audio_root, subject_id)
        if fpath is None:
            continue
        for fn in list_audio_files(fpath):
            base = os.path.splitext(fn)[0]
            out_path = os.path.join(save_path, f"{subject_id}_{base}_ASR.txt")
            if not os.path.exists(out_path):
                missing_n += 1
                subjects_hit.add(subject_id)
    return missing_n, subjects_hit


def make_key(subject_id: str, filename: str) -> str:

    wavstem = os.path.splitext(str(filename))[0]
    return f"{str(subject_id)}_{wavstem}"

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def transcribe_one(model, audio_path: str, language: Optional[str], task: str) -> Tuple[str, float]:
    result = model.transcribe(
        audio_path,
        language=language if language else None,
        task=task,
        fp16=ASR_FP16,
        beam_size=ASR_BEAM_SIZE,
        best_of=ASR_BEST_OF,
        temperature=ASR_TEMPERATURE,
    )
    text = clean_text(result.get("text", ""))
    avg_logprob = float(result.get("avg_logprob", float("nan")))
    return text, avg_logprob

def process_subjects_asr(subject_ids: List[str], save_path: str, class_name: str, model) -> pd.DataFrame:

    rows = []
    processed = 0

    for subject_id in subject_ids:
        fpath = find_subject_folder(audio_root, subject_id)
        if fpath is None:
            if VERBOSE_ASR:
                print(f"  No audio folder for subject {subject_id} under {audio_root}")
            continue

        wavefiles = list_audio_files(fpath)
        if not wavefiles:
            if VERBOSE_ASR:
               print(f"  No audio files found in {fpath}")
            continue

        for fn in wavefiles:
            audio_path = os.path.join(fpath, fn)
            base = os.path.splitext(fn)[0]

            out_name = f"{subject_id}_{base}_ASR.txt"
            out_path = os.path.join(save_path, out_name)

            try:
                # 缓存命中：已有转写文本则直接复用，不再调用 Whisper
                if SKIP_IF_EXISTS and (not FORCE_RETRANSCRIBE) and os.path.exists(out_path):
                    with open(out_path, "r", encoding="utf-8") as f:
                        text = clean_text(f.read())
                    rows.append({
                        "class": class_name,
                        "subject_id": subject_id,
                        "file": fn,
                        "audio_path": audio_path,
                        "text_path": out_path,
                        "transcript": text,
                        "avg_logprob": float("nan"),
                        "cached": 1,
                    })
                    if PRINT_ASR_FILE_LOG:
                        print(f"[ASR CACHE] {class_name} {subject_id} | {fn} -> {out_name}")
                    continue

                if model is None:
                    if VERBOSE_ASR:
                        print(f"[ASR MISS] Skip missing cache: {class_name} {subject_id} | {fn}")
                    continue

                text, avg_logprob = transcribe_one(model, audio_path, LANGUAGE, TASK)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text + "\n")

                rows.append({
                    "class": class_name,
                    "subject_id": subject_id,
                    "file": fn,
                    "audio_path": audio_path,
                    "text_path": out_path,
                    "transcript": text,
                    "avg_logprob": avg_logprob,
                    "cached": 0,
                })
                processed += 1
                if PRINT_ASR_FILE_LOG:
                    print(f"[ASR OK] {class_name} {subject_id} | {fn} -> {out_name}")

            except Exception as e:
                if PRINT_ASR_FILE_LOG or VERBOSE_ASR:
                    print(f"[ASR FAIL] {class_name} {subject_id} | {fn} | error={e}")
                rows.append({
                    "class": class_name,
                    "subject_id": subject_id,
                    "file": fn,
                    "audio_path": audio_path,
                    "text_path": out_path,
                    "transcript": "",
                    "avg_logprob": float("nan"),
                    "error": str(e),
                })


    return pd.DataFrame(rows)


# =========================
# 二分类（BERT768 -> 单模型 MLP）
# =========================
class FeatDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TextDS(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: np.ndarray):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
        )
        self.out = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.in_proj(x)
        h = h + self.block(h)
        h = torch.nn.functional.gelu(h)
        return self.out(h)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x):
        return self.fc(x)


def build_classifier(classifier_type: str) -> nn.Module:
    t = str(classifier_type).strip().lower()
    if t == "linear":
        return LinearHead()
    if t == "mlp":
        return ResidualMLP()
    raise ValueError(f"Unsupported classifier_type={classifier_type}, expected 'linear' or 'mlp'")


def zscore_by_train(X_tr: np.ndarray, X_va: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_tr.mean(axis=0, keepdims=True)
    sigma = X_tr.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return (X_tr - mu) / sigma, (X_va - mu) / sigma


def fit_one_split(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    device: str,
    classifier_type: str,
    lr: float,
    weight_decay: float,
) -> Tuple[nn.Module, float]:
    class_weight = None
    if USE_CLASS_WEIGHT:
        cls_counts = np.bincount(y_tr, minlength=2).astype(np.float32)
        total = float(cls_counts.sum())
        w0 = total / (2.0 * max(cls_counts[0], 1.0))
        w1 = total / (2.0 * max(cls_counts[1], 1.0))
        class_weight = torch.tensor([w0, w1], dtype=torch.float32)

    X_tr_use, X_va_use = X_tr, X_va
    if classifier_type == "linear":
        # 线性头对尺度更敏感，使用训练集统计做标准化通常更稳。
        X_tr_use, X_va_use = zscore_by_train(X_tr, X_va)

    model = build_classifier(classifier_type).to(device)
    train_loader = DataLoader(FeatDS(X_tr_use, y_tr), batch_size=CLS_BATCH, shuffle=True)
    val_loader = DataLoader(FeatDS(X_va_use, y_va), batch_size=CLS_BATCH, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device) if class_weight is not None else None)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=3)

    best_acc = -1.0
    best_state = None
    no_improve = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        train_preds, train_golds = [], []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            p_tr = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            train_preds.extend(p_tr)
            train_golds.extend(yb.detach().cpu().numpy().tolist())
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                p = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds.extend(p)
                golds.extend(yb.numpy().tolist())
        val_acc = accuracy_score(golds, preds)
        train_acc = accuracy_score(train_golds, train_preds) if train_golds else 0.0
        scheduler.step(val_acc)
        print(f"[{classifier_type}] Epoch {ep}/{EPOCHS} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if PATIENCE > 0 and no_improve >= PATIENCE:
                print(f"[{classifier_type}] Early stop at epoch {ep} (patience={PATIENCE})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_acc)


def train_classifier(
    feats: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    输入：feats (N,768), labels (N,)
    输出：pred (N,), prob_mdd (N,), model
    """
    set_seed(SEED)

    idx_all = np.arange(len(labels))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    idx_tr, idx_te = next(gss.split(idx_all, labels, groups=groups))

    X_tr, y_tr = feats[idx_tr], labels[idx_tr]
    X_te, y_te = feats[idx_te], labels[idx_te]
    grp_tr = groups[idx_tr]
    grp_te = groups[idx_te]
    print(f"Classifier type setting: {CLASSIFIER_TYPE}")
    print(f"Train size={len(idx_tr)}, Val size={len(idx_te)}")
    print(f"Unique subjects train={len(np.unique(grp_tr))}, val={len(np.unique(grp_te))}")
    overlap = set(np.unique(grp_tr)).intersection(set(np.unique(grp_te)))
    print(f"Subject overlap train/val: {len(overlap)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidates = []
    cfg = str(CLASSIFIER_TYPE).strip().lower()
    if cfg == "auto":
        for lr in LINEAR_LR_CANDIDATES:
            for wd in WD_CANDIDATES:
                candidates.append(("linear", lr, wd))
        for lr in MLP_LR_CANDIDATES:
            for wd in WD_CANDIDATES:
                candidates.append(("mlp", lr, wd))
    elif cfg == "linear":
        for lr in LINEAR_LR_CANDIDATES:
            for wd in WD_CANDIDATES:
                candidates.append(("linear", lr, wd))
    elif cfg == "mlp":
        for lr in MLP_LR_CANDIDATES:
            for wd in WD_CANDIDATES:
                candidates.append(("mlp", lr, wd))
    else:
        raise ValueError("CLASSIFIER_TYPE must be one of: linear / mlp / auto")

    best_model, best_acc, best_name, best_lr, best_wd = None, -1.0, "", 0.0, 0.0
    for name, lr, wd in candidates:
        model_try, acc_try = fit_one_split(
            X_tr,
            y_tr,
            X_te,
            y_te,
            device,
            classifier_type=name,
            lr=lr,
            weight_decay=wd,
        )
        print(f"[Try] head={name} lr={lr:.1e} wd={wd:.1e} | val_acc={acc_try:.4f}")
        if acc_try > best_acc:
            best_acc = acc_try
            best_model = model_try
            best_name = name
            best_lr = lr
            best_wd = wd

    model, best_acc_refit = fit_one_split(
        X_tr,
        y_tr,
        X_te,
        y_te,
        device,
        classifier_type=best_name,
        lr=best_lr,
        weight_decay=best_wd,
    )
    best_acc = float(best_acc_refit)
    print(f"Selected head: {best_name} | lr={best_lr:.1e} wd={best_wd:.1e}")
    print(f"Total Accuracy: {best_acc:.4f}")

    # 全量推理（用于兼容原有下游变量）
    model.eval()
    logits_all = []
    with torch.no_grad():
        for i in range(0, feats.shape[0], CLS_BATCH):
            xb = torch.tensor(feats[i:i + CLS_BATCH], dtype=torch.float32, device=device)
            logits_all.append(model(xb).cpu().numpy())
    logits_all = np.vstack(logits_all)
    prob = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
    pred = np.argmax(prob, axis=1)
    prob_mdd = prob[:, 1]

    return pred, prob_mdd, model


def resolve_finetune_model_name() -> str:
    for name in FINETUNE_MODEL_CANDIDATES:
        if os.path.exists(name):
            return name
    return FINETUNE_MODEL_CANDIDATES[0]


def set_backbone_trainable(model: nn.Module, trainable: bool):
    base = getattr(model, "base_model", None)
    if base is None:
        return
    for p in base.parameters():
        p.requires_grad = trainable


def _finetune_one_seed(
    texts: List[str],
    labels: np.ndarray,
    groups: np.ndarray,
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray, object, float, int]:
    """单次 group split + 训练，返回 (pred, prob_mdd, model, best_val_acc, best_epoch)。"""
    set_seed(split_seed)
    idx_all = np.arange(len(labels))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=split_seed)
    idx_tr, idx_te = next(gss.split(idx_all, labels, groups=groups))

    texts_tr = [texts[i] for i in idx_tr]
    texts_te = [texts[i] for i in idx_te]
    y_tr, y_te = labels[idx_tr], labels[idx_te]

    model_name = resolve_finetune_model_name()
    local_only = os.path.exists(model_name)
    print(f"Finetune backbone: {model_name} | local_files_only={local_only}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        local_files_only=local_only,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    enc_tr = tokenizer(
        texts_tr, truncation=True, padding=True, max_length=FT_MAX_LEN, return_tensors="pt"
    )
    enc_te = tokenizer(
        texts_te, truncation=True, padding=True, max_length=FT_MAX_LEN, return_tensors="pt"
    )
    gen = torch.Generator()
    gen.manual_seed(split_seed)
    tr_loader = DataLoader(
        TextDS(enc_tr, y_tr), batch_size=FT_BATCH, shuffle=True, generator=gen
    )
    te_loader = DataLoader(TextDS(enc_te, y_te), batch_size=FT_BATCH, shuffle=False)

    class_weight = None
    if USE_CLASS_WEIGHT:
        cls_counts = np.bincount(y_tr, minlength=2).astype(np.float32)
        total = float(cls_counts.sum())
        w0 = total / (2.0 * max(cls_counts[0], 1.0))
        w1 = total / (2.0 * max(cls_counts[1], 1.0))
        class_weight = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    loss_kw = {}
    if class_weight is not None:
        loss_kw["weight"] = class_weight
    if FT_LABEL_SMOOTHING > 0:
        loss_kw["label_smoothing"] = float(FT_LABEL_SMOOTHING)
    criterion = nn.CrossEntropyLoss(**loss_kw)

    optim = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    scheduler = None
    if FT_SCHEDULER_MODE == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="max",
            factor=FT_PLATEAU_FACTOR,
            patience=FT_PLATEAU_PATIENCE,
            min_lr=FT_MIN_LR,
        )

    best_acc = -1.0
    best_ep = 0
    best_state = None
    no_improve = 0
    for ep in range(1, FT_EPOCHS + 1):
        freeze_now = ep <= FT_FREEZE_EPOCHS
        set_backbone_trainable(model, not freeze_now)
        model.train()
        for batch in tr_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(out.logits, batch["labels"])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FT_GRAD_CLIP)
            optim.step()

        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch in te_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                p = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds.extend(p)
                golds.extend(batch["labels"].cpu().numpy().tolist())
        val_acc = accuracy_score(golds, preds)
        if scheduler is not None:
            scheduler.step(val_acc)
        cur_lr = float(optim.param_groups[0]["lr"])
        stage = "head-only" if freeze_now else "full"
        print(
            f"[finetune] seed={split_seed} Epoch {ep}/{FT_EPOCHS} | stage={stage} "
            f"| lr={cur_lr:.2e} | val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= FT_PATIENCE:
                print(f"[finetune] Early stop at epoch {ep} (patience={FT_PATIENCE})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[finetune] seed={split_seed} Best val_acc={best_acc:.4f} at epoch {best_ep}")

    model.eval()
    enc_all = tokenizer(
        texts, truncation=True, padding=True, max_length=FT_MAX_LEN, return_tensors="pt"
    )
    all_loader = DataLoader(TextDS(enc_all, labels), batch_size=FT_BATCH, shuffle=False)
    logits_all = []
    with torch.no_grad():
        for batch in all_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            logits_all.append(logits.cpu().numpy())
    logits_all = np.vstack(logits_all)
    prob = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
    pred = np.argmax(prob, axis=1)
    prob_mdd = prob[:, 1]
    return pred, prob_mdd, model, best_acc, best_ep


def train_text_finetune(
    texts: List[str],
    labels: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, object]:
    seeds = list(FT_EVAL_SEEDS) if FT_MULTI_SEED_EVAL else [SEED]
    accs: List[float] = []
    best_pred = best_prob = best_model = None
    best_acc_overall = -1.0
    best_seed_used = SEED

    for s in seeds:
        pred, prob_mdd, model, best_val, _ep = _finetune_one_seed(texts, labels, groups, s)
        accs.append(best_val)
        if best_val > best_acc_overall:
            best_acc_overall = best_val
            best_pred, best_prob, best_model = pred, prob_mdd, model
            best_seed_used = s
        print(f"\n--- seed={s} done | best val_acc={best_val:.4f} ---\n")

    if FT_MULTI_SEED_EVAL and len(accs) > 1:
        print(
            f"[finetune] Multi-seed summary: max={max(accs):.4f} "
            f"mean={float(np.mean(accs)):.4f} std={float(np.std(accs)):.4f} "
            f"(seeds={seeds})"
        )
    print(f"Total Accuracy (best val across runs): {best_acc_overall:.4f} | best seed={best_seed_used}")
    assert best_pred is not None and best_model is not None
    return best_pred, best_prob, best_model


# =========================
# 主流程
# =========================
def main():
    mdd_sub = list_subject_ids_from_audio_root(audio_root, "0201")
    normal_sub: List[str] = []
    for _hc_prefix in ("0202", "0203"):
        normal_sub.extend(list_subject_ids_from_audio_root(audio_root, _hc_prefix))
    normal_sub = sorted(set(normal_sub))

    print(f"MDD subjects: {len(mdd_sub)}")
    print(f"Control subjects: {len(normal_sub)}")
    print(f"Audio root: {audio_root}")
    print(f"Save root:  {save_base_path}")
    print(f"ASR: whisper model={WHISPER_MODEL}, language={LANGUAGE}, task={TASK}")
    if FORCE_RETRANSCRIBE:
        print("ASR mode: force re-transcribe all files")
    else:
        print("ASR mode: reuse existing txt when available")

    out_npy = os.path.join(save_base_path, "bert768_features.npy")

    # 2) ASR：已有 txt 则跳过；若有音频尚未生成 txt，必须加载 Whisper（否则会永远缺 14 人那种情况）
    miss_m, sub_m = scan_missing_asr_txt(mdd_sub, save_path_mdd)
    miss_c, sub_c = scan_missing_asr_txt(normal_sub, save_path_control)
    missing_txt_total = miss_m + miss_c
    if missing_txt_total > 0:
        print(
            f"ASR: {missing_txt_total} transcript file(s) missing "
            f"({len(sub_m)} MDD + {len(sub_c)} Control subjects affected); will load Whisper."
        )
    need_asr_model = FORCE_RETRANSCRIBE or (not SKIP_IF_EXISTS) or (missing_txt_total > 0)
    model_asr = whisper.load_model(WHISPER_MODEL) if need_asr_model else None
    if not need_asr_model:
        print("ASR model loading skipped: all expected *_ASR.txt already exist.")
    df_mdd = process_subjects_asr(mdd_sub, save_path_mdd, "MDD", model_asr)
    df_ctl = process_subjects_asr(normal_sub, save_path_control, "Control", model_asr)
    df_all = pd.concat([df_mdd, df_ctl], ignore_index=True)


    # 3) 过滤空转写 + 打标签
    df2 = df_all.copy()
    df2["transcript"] = df2["transcript"].astype(str)
    df2 = df2[df2["transcript"].str.len() > 0].reset_index(drop=True)

    df2["label"] = df2["class"].map(lambda x: 1 if str(x).upper() == "MDD" else 0).astype(int)



    labels = df2["label"].values.astype(int)
    groups = df2["subject_id"].astype(str).values
    pred = prob_mdd = None

    if TRAIN_MODE == "finetune":
        if RUN_CLASSIFIER_TRAIN:
            pred, prob_mdd, _ = train_text_finetune(df2["transcript"].tolist(), labels, groups)
        else:
            print("RUN_CLASSIFIER_TRAIN=False: skip BERT finetune / depression classification.")
    else:
        # 4) BERT 768 特征（优先复用缓存）
        feats = None
        if REUSE_CACHED_FEATURES and os.path.exists(out_npy):
            feats = np.load(out_npy)

        if feats is None:
            encoder = Bert768Encoder(
                model_name=r"D:\models\bert-base-chinese",
            )
            feats = encoder.encode_768(
                df2["transcript"].tolist(),
                max_len=MAX_LEN,
                batch_size=FEAT_BATCH,
                normalize=L2_NORMALIZE,
            )

        # 导出 keyed 特征：不做任何 split，直接保存全体样本特征用于多模态融合。
        if EXPORT_BERT_FEATS_NPZ:
            keys = np.array(
                [
                    make_key(sid, fn)
                    for sid, fn in zip(
                        df2["subject_id"].astype(str).tolist(),
                        df2["file"].astype(str).tolist(),
                    )
                ],
                dtype=object,
            )
            np.savez(
                BERT_FEATS_NPZ_PATH,
                keys=keys,
                feat=np.asarray(feats, dtype=np.float32),
                label=labels.astype(np.int64),
                subject_id=df2["subject_id"].astype(str).values.astype(object),
                wav_file=df2["file"].astype(str).values.astype(object),
            )
            print(f"Saved keyed BERT feats: {BERT_FEATS_NPZ_PATH}")

        if not MINIMAL_OUTPUT:
            np.save(out_npy, feats.astype(np.float32))

        if RUN_CLASSIFIER_TRAIN:
            pred, prob_mdd, _ = train_classifier(feats, labels, groups)
        else:
            print("RUN_CLASSIFIER_TRAIN=False: skip linear/MLP classifier training.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
