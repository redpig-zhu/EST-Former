# -*- coding: utf-8 -*-
import os
import argparse
import random
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pretrained_binary_model(freeze_backbone=True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier."):
                p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
    )
    return model


def stratified_split_indices(targets, val_split=DEFAULT_VAL_SPLIT, seed=DEFAULT_SEED):
    rng = random.Random(seed)
    by_class = {}
    for i, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(i)

    train_indices = []
    val_indices = []
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n < 2:
            raise ValueError(f"Class {cls} has too few samples ({n}). Need at least 2 per class.")
        n_val = max(int(round(n * val_split)), 1)
        n_val = min(n_val, n - 1)
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def compute_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_logits = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach())
        all_targets.append(labels.detach())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    all_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, 2, device=device)
    all_targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, dtype=torch.long, device=device)
    acc = compute_accuracy(all_logits, all_targets)
    return epoch_loss, acc


def train_pretrained_binary_classifier(
    image_root,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    val_split=DEFAULT_VAL_SPLIT,
    seed=DEFAULT_SEED,
    freeze_backbone=True,
    out_dir=None,
):
    if not os.path.exists(image_root):
        raise FileNotFoundError(f"image_root not found: {image_root}")

    set_seed(seed)
    device = get_device()
    print(f"\nTraining on device: {device}")

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train_dataset = datasets.ImageFolder(root=image_root, transform=train_tf)
    full_eval_dataset = datasets.ImageFolder(root=image_root, transform=eval_tf)
    class_names = full_train_dataset.classes
    if len(class_names) != 2:
        raise ValueError(f"Expected 2 classes, got {len(class_names)}: {class_names}")

    targets = full_train_dataset.targets
    train_indices, val_indices = stratified_split_indices(targets, val_split=val_split, seed=seed)
    train_set = Subset(full_train_dataset, train_indices)
    val_set = Subset(full_eval_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_pretrained_binary_model(freeze_backbone=freeze_backbone).to(device)

    train_targets = [targets[i] for i in train_indices]
    train_count = Counter(train_targets)
    total_train = len(train_targets)
    class_weights = []
    for cls_idx in range(len(class_names)):
        cls_n = train_count.get(cls_idx, 0)
        class_weights.append(total_train / max(cls_n, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    best_val_acc = -1.0
    best_state = None

    full_count = Counter(targets)
    val_targets = [targets[i] for i in val_indices]
    val_count = Counter(val_targets)

    print(f"Classes: {class_names}")
    print(f"Model: efficientnet_b0 | freeze_backbone={freeze_backbone}")
    print(f"Class count (all): {dict(full_count)}")
    print(f"Class count (train): {dict(train_count)}")
    print(f"Class count (val): {dict(val_count)}")
    print(f"Samples: train={len(train_set)}, val={len(val_set)}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.4f}"
        )

    save_dir = out_dir if out_dir else os.path.dirname(image_root)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "eeg2d_binary_best.pth")
    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "class_to_idx": full_train_dataset.class_to_idx,
                "best_val_acc": float(best_val_acc),
                "image_root": image_root,
                "model_name": "efficientnet_b0",
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            ckpt_path,
        )

    print("=" * 70)
    print(f"Final classification accuracy (val): {best_val_acc:.4f}")


def parse_args():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_image_root = os.path.join(repo_root, "multimodal_overlap", "EEG2Dimages_paper224")
    default_out_dir = os.path.join(repo_root, "checkpoints")

    parser = argparse.ArgumentParser(description="Train pretrained binary classifier on EEG 2D images")
    parser.add_argument("--image-root", type=str, default=default_image_root)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=str, default=default_out_dir)
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train classifier head")
    parser.add_argument("--no-freeze-backbone", action="store_true", help="Train full network")
    return parser.parse_args()


def main():
    args = parse_args()
    freeze_backbone = True
    if args.no_freeze_backbone:
        freeze_backbone = False
    elif args.freeze_backbone:
        freeze_backbone = True

    train_pretrained_binary_classifier(
        image_root=args.image_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        freeze_backbone=freeze_backbone,
        out_dir=args.out_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
