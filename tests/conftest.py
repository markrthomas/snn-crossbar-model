"""Shared pytest fixtures for accuracy/model tests.

All fixtures use synthetic data so tests never require an MNIST download.
Model dimensions are kept tiny (hidden_dim=8, num_steps=2) for speed.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.crossbar_snn import CrossbarConfig, CrossbarSNN


@pytest.fixture
def tiny_cfg() -> CrossbarConfig:
    return CrossbarConfig(
        hidden_dim=8,
        num_steps=2,
        weight_levels=4,
        crossbar_rows=16,
        crossbar_cols=16,
    )


@pytest.fixture
def tiny_model(tiny_cfg: CrossbarConfig) -> CrossbarSNN:
    torch.manual_seed(0)
    model = CrossbarSNN(tiny_cfg)
    model.eval()
    return model


@pytest.fixture
def tiny_loader() -> DataLoader:
    """16 MNIST-shaped samples with random labels."""
    torch.manual_seed(1)
    images = torch.rand(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    return DataLoader(TensorDataset(images, labels), batch_size=8)
