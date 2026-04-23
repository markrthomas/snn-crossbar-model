"""Tests for src/train_utils.py"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.crossbar_snn import CrossbarConfig, CrossbarSNN
from src.train_utils import evaluate, train_one_epoch


def test_evaluate_returns_float_in_unit_interval(tiny_model, tiny_loader):
    device = torch.device("cpu")
    acc = evaluate(tiny_model, tiny_loader, device)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_evaluate_perfect_accuracy():
    """A model whose fc2 weights force a single dominant class should reach acc=1
    on a dataset where every label matches that class."""
    cfg = CrossbarConfig(hidden_dim=8, num_steps=2, weight_levels=4,
                         crossbar_rows=16, crossbar_cols=16)
    model = CrossbarSNN(cfg)
    # Drive output dim 0 to always win by setting its fc2 row to large values.
    with torch.no_grad():
        model.fc2.weight.data.fill_(0.0)
        model.fc2.weight.data[0, :] = 1.0   # class 0 always accumulates the most spikes

    images = torch.rand(16, 1, 28, 28)
    labels = torch.zeros(16, dtype=torch.long)  # all label 0
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    acc = evaluate(model, loader, torch.device("cpu"))
    assert acc == pytest.approx(1.0)


def test_evaluate_empty_loader_returns_zero():
    """DataLoader with no batches should return 0.0 without crashing."""
    cfg = CrossbarConfig(hidden_dim=8, num_steps=2, weight_levels=4,
                         crossbar_rows=16, crossbar_cols=16)
    model = CrossbarSNN(cfg)
    empty_loader: DataLoader = DataLoader(TensorDataset(
        torch.empty(0, 1, 28, 28), torch.empty(0, dtype=torch.long)
    ), batch_size=8)
    acc = evaluate(model, empty_loader, torch.device("cpu"))
    assert acc == 0.0


def test_train_one_epoch_returns_finite_loss_and_acc(tiny_cfg, tiny_loader):
    torch.manual_seed(42)
    model = CrossbarSNN(tiny_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    loss, acc = train_one_epoch(model, tiny_loader, optimizer, criterion,
                                torch.device("cpu"), epoch=1)

    assert torch.isfinite(torch.tensor(loss)), f"loss={loss} is not finite"
    assert torch.isfinite(torch.tensor(acc)), f"acc={acc} is not finite"
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_updates_weights(tiny_cfg, tiny_loader):
    """Weights must change after one training step."""
    torch.manual_seed(7)
    model = CrossbarSNN(tiny_cfg)
    w_before = model.fc1.weight.data.clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    train_one_epoch(model, tiny_loader, optimizer, criterion,
                    torch.device("cpu"), epoch=1)

    assert not torch.allclose(model.fc1.weight.data, w_before), \
        "fc1.weight did not change after a training epoch"


def test_train_one_epoch_raises_on_nan_loss(tiny_cfg):
    """Non-finite loss must raise RuntimeError immediately.

    Use a mock criterion that unconditionally returns NaN to test the guard
    directly, rather than relying on NaN propagation through the spike model
    (which can be suppressed by all-zero sparse inputs).
    """
    class _NanLoss(torch.nn.Module):
        def forward(self, logits, targets):
            return torch.tensor(float("nan"))

    model = CrossbarSNN(tiny_cfg)
    images = torch.rand(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(RuntimeError, match="Non-finite loss"):
        train_one_epoch(model, loader, optimizer, _NanLoss(),
                        torch.device("cpu"), epoch=1)
