"""Tests for noise-aware QAT: QuantLinear noise injection and sweep checkpoint saving."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep_qat import run_config
from src.crossbar_snn import CrossbarSNN, QuantLinear
from src.train_utils import evaluate


# ---------------------------------------------------------------------------
# QuantLinear noise injection
# ---------------------------------------------------------------------------

def test_quant_linear_no_noise_in_eval_mode():
    """noise_sigma > 0 must have no effect when model is in eval mode."""
    torch.manual_seed(0)
    layer = QuantLinear(4, 4, levels=8, noise_sigma=0.5)
    layer.eval()

    x = torch.rand(2, 4)
    out1 = layer(x)
    out2 = layer(x)
    assert torch.allclose(out1, out2), "eval-mode output should be deterministic despite noise_sigma > 0"


def test_quant_linear_noise_active_in_train_mode():
    """noise_sigma > 0 must produce different outputs across calls in train mode."""
    torch.manual_seed(0)
    layer = QuantLinear(16, 16, levels=8, noise_sigma=0.5)
    layer.train()

    x = torch.rand(4, 16)
    out1 = layer(x).detach()
    out2 = layer(x).detach()
    assert not torch.allclose(out1, out2), \
        "train-mode outputs should differ across calls when noise_sigma > 0"


def test_quant_linear_zero_noise_deterministic_in_train_mode():
    """noise_sigma=0 must give identical outputs even in train mode."""
    layer = QuantLinear(8, 8, levels=8, noise_sigma=0.0)
    layer.train()

    x = torch.rand(2, 8)
    out1 = layer(x).detach()
    out2 = layer(x).detach()
    assert torch.allclose(out1, out2), "zero-noise train outputs should be deterministic"


def test_quant_linear_noise_does_not_corrupt_weights():
    """Weight parameter must not be modified by the noise forward pass."""
    layer = QuantLinear(8, 8, levels=8, noise_sigma=1.0)
    layer.train()
    w_before = layer.weight.data.clone()

    x = torch.rand(2, 8)
    layer(x)  # forward with noise

    assert torch.allclose(layer.weight.data, w_before), \
        "noise forward pass must not modify the stored weight parameter"


# ---------------------------------------------------------------------------
# CrossbarSNN.set_training_noise
# ---------------------------------------------------------------------------

def test_set_training_noise_sets_both_layers(tiny_model):
    tiny_model.set_training_noise(0.3)
    assert tiny_model.fc1.noise_sigma == pytest.approx(0.3)
    assert tiny_model.fc2.noise_sigma == pytest.approx(0.3)


def test_set_training_noise_zero_clears_noise(tiny_model):
    tiny_model.set_training_noise(0.5)
    tiny_model.set_training_noise(0.0)
    assert tiny_model.fc1.noise_sigma == 0.0
    assert tiny_model.fc2.noise_sigma == 0.0


def test_noise_aware_training_changes_gradient_path(tiny_cfg):
    """Gradients should flow to fc1.weight even with noise enabled."""
    model = CrossbarSNN(tiny_cfg)
    model.set_training_noise(0.2)
    model.train()

    images = torch.rand(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    logits, _ = model(images)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert model.fc1.weight.grad is not None, "fc1.weight.grad should not be None"
    assert torch.any(model.fc1.weight.grad != 0), "fc1.weight.grad should be non-zero"


def test_noise_aware_training_stays_finite(tiny_cfg, tiny_loader):
    """Noise-aware training must produce finite losses and valid accuracy throughout.

    Whether noise-aware training actually improves robustness is a property
    that only manifests at realistic model/dataset scale; this test just
    verifies the mechanism doesn't blow up or produce NaN.
    """
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = CrossbarSNN(tiny_cfg)
    model.set_training_noise(0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for imgs, lbls in tiny_loader:
            opt.zero_grad()
            logits, _ = model(imgs)
            loss = crit(logits, lbls)
            assert torch.isfinite(loss), f"Non-finite loss at epoch {epoch}: {loss.item()}"
            loss.backward()
            assert all(torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.grad is not None), \
                "Non-finite gradient during noise-aware training"
            opt.step()

    acc = evaluate(model, tiny_loader, device)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# sweep_qat.run_config — checkpoint saving and cosine schedule
# ---------------------------------------------------------------------------

@pytest.fixture
def sweep_loaders():
    torch.manual_seed(2)
    images = torch.rand(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    ds = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=8)
    return loader, loader  # train == test for this fixture


def test_run_config_saves_checkpoint(tiny_cfg, sweep_loaders):
    train_loader, test_loader = sweep_loaders
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp)
        run_config(tiny_cfg, train_loader, test_loader, torch.device("cpu"),
                   epochs=1, lr=1e-3, lr_schedule="none",
                   noise_sigma=0.0, save_dir=save_dir)
        tag = f"wl{tiny_cfg.weight_levels}_ns{tiny_cfg.num_steps}_hd{tiny_cfg.hidden_dim}"
        assert (save_dir / f"{tag}.pt").exists(), "checkpoint .pt file not created"
        assert (save_dir / f"{tag}_cfg.json").exists(), "config .json file not created"


def test_run_config_checkpoint_cfg_json_matches(tiny_cfg, sweep_loaders):
    train_loader, test_loader = sweep_loaders
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp)
        run_config(tiny_cfg, train_loader, test_loader, torch.device("cpu"),
                   epochs=1, lr=1e-3, lr_schedule="none",
                   noise_sigma=0.05, save_dir=save_dir)
        tag = f"wl{tiny_cfg.weight_levels}_ns{tiny_cfg.num_steps}_hd{tiny_cfg.hidden_dim}"
        cfg_data = json.loads((save_dir / f"{tag}_cfg.json").read_text())
        assert cfg_data["weight_levels"] == tiny_cfg.weight_levels
        assert cfg_data["hidden_dim"] == tiny_cfg.hidden_dim
        assert cfg_data["noise_sigma_training"] == pytest.approx(0.05)


def test_run_config_no_save_dir_creates_no_files(tiny_cfg, sweep_loaders):
    train_loader, test_loader = sweep_loaders
    with tempfile.TemporaryDirectory() as tmp:
        result = run_config(tiny_cfg, train_loader, test_loader, torch.device("cpu"),
                            epochs=1, lr=1e-3, lr_schedule="none",
                            noise_sigma=0.0, save_dir=None)
        assert result["best_test_acc"] >= 0.0
        # No .pt files should have been written anywhere
        assert list(Path(tmp).glob("*.pt")) == []


def test_run_config_result_keys(tiny_cfg, sweep_loaders):
    train_loader, test_loader = sweep_loaders
    result = run_config(tiny_cfg, train_loader, test_loader, torch.device("cpu"),
                        epochs=1, lr=1e-3, lr_schedule="cosine",
                        noise_sigma=0.0, save_dir=None)
    for key in ("weight_levels", "num_steps", "hidden_dim", "best_test_acc",
                "final_test_acc", "tile_count", "bits_per_weight", "epochs"):
        assert key in result, f"missing key '{key}' in run_config result"


def test_run_config_cosine_schedule_runs_without_error(tiny_cfg, sweep_loaders):
    train_loader, test_loader = sweep_loaders
    result = run_config(tiny_cfg, train_loader, test_loader, torch.device("cpu"),
                        epochs=2, lr=1e-3, lr_schedule="cosine",
                        noise_sigma=0.0, save_dir=None)
    assert len(result["epochs"]) == 2
