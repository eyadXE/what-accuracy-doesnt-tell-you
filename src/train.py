"""Training loop with checkpointing, seeding, and per-epoch metric logging."""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import nll as nll_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    all_logits, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    return dict(
        loss=total_loss / n,
        acc=total_correct / n,
        nll=nll_fn(logits_np, labels_np),
        logits=logits_np,
        labels=labels_np,
    )


def train_model(
    model: nn.Module,
    train_ds,
    val_ds,
    device,
    epochs: int,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    seed: int = 0,
    checkpoint_path: str | None = None,
    log_every_epochs: int = 1,
    scheduler_name: str = "cosine",
    num_workers: int = 0,
):
    """Train `model`, return (model, history) where history is a list of per-epoch dicts.

    If `checkpoint_path` exists on disk, loads it and skips training (cache hit).
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        return model, ckpt["history"]

    set_seed(seed)
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = []
    t0 = time.time()
    for epoch in tqdm(range(epochs), desc=checkpoint_path or "train"):
        model.train()
        total_loss, total_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        if scheduler is not None:
            scheduler.step()

        train_loss, train_acc = total_loss / n, total_correct / n
        val_metrics = evaluate(model, val_loader, device, criterion)
        row = dict(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["acc"],
            val_nll=val_metrics["nll"],
        )
        history.append(row)
        if epoch % log_every_epochs == 0 or epoch == epochs - 1:
            tqdm.write(
                f"[{checkpoint_path or 'model'}] epoch {epoch:3d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"val_nll={val_metrics['nll']:.4f}"
            )

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_path)

    return model, history


@torch.no_grad()
def get_logits(model, dataset, device, batch_size: int = 512, num_workers: int = 0):
    """Run inference, return (logits, labels) as numpy arrays. labels are -1 for OOD data."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(np.asarray(y))
    return np.concatenate(all_logits), np.concatenate(all_labels)
