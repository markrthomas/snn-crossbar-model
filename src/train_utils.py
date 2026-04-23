"""Shared training and evaluation utilities used by train.py and sweep scripts."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.crossbar_snn import CrossbarSNN


def train_one_epoch(
    model: CrossbarSNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (train_loss, train_acc)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_items += labels.numel()

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def evaluate(model: CrossbarSNN, loader: DataLoader, device: torch.device) -> float:
    """Return top-1 accuracy on loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)
