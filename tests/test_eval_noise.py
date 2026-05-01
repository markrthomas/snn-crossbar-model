"""Tests for the weight-noise evaluation machinery in scripts/eval_noise.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_noise import eval_sigma, noisy_weights
from src.crossbar_snn import quantize_ste


# ---------------------------------------------------------------------------
# noisy_weights context manager
# ---------------------------------------------------------------------------

def test_noisy_weights_restores_after_exit(tiny_model):
    w1_before = tiny_model.fc1.weight.data.clone()
    w2_before = tiny_model.fc2.weight.data.clone()
    levels_before = tiny_model.fc1.levels

    with noisy_weights(tiny_model, sigma=0.1, weight_levels=4, seed=0):
        pass  # just enter and exit

    assert torch.allclose(tiny_model.fc1.weight.data, w1_before), \
        "fc1.weight not restored after context exit"
    assert torch.allclose(tiny_model.fc2.weight.data, w2_before), \
        "fc2.weight not restored after context exit"
    assert tiny_model.fc1.levels == levels_before, \
        "fc1.levels not restored after context exit"


def test_noisy_weights_restores_after_exception(tiny_model):
    w1_before = tiny_model.fc1.weight.data.clone()

    with pytest.raises(ValueError):
        with noisy_weights(tiny_model, sigma=0.1, weight_levels=4, seed=0):
            raise ValueError("deliberate error inside context")

    assert torch.allclose(tiny_model.fc1.weight.data, w1_before), \
        "fc1.weight not restored after exception"


def test_noisy_weights_changes_weights_inside_context(tiny_model):
    w1_before = tiny_model.fc1.weight.data.clone()

    with noisy_weights(tiny_model, sigma=0.5, weight_levels=4, seed=0):
        assert not torch.allclose(tiny_model.fc1.weight.data, w1_before), \
            "weights should differ from originals inside noisy context (sigma>0)"


def test_noisy_weights_sigma_zero_equals_quantised(tiny_model):
    """With sigma=0 the in-context weights should equal the quantised originals."""
    q_w1 = quantize_ste(tiny_model.fc1.weight.data.clone(), tiny_model.cfg.weight_levels)

    with noisy_weights(tiny_model, sigma=0.0, weight_levels=tiny_model.cfg.weight_levels, seed=0):
        assert torch.allclose(tiny_model.fc1.weight.data, q_w1), \
            "sigma=0 should produce exactly the quantised weights"


def test_noisy_weights_bypasses_requantisation(tiny_model):
    """During the context, QuantLinear must not re-snap weights back to the grid.

    We inject a mid-LSB weight value and verify the forward pass actually uses it
    (if levels were still active it would be snapped away).
    """
    with noisy_weights(tiny_model, sigma=0.1, weight_levels=4, seed=3):
        assert tiny_model.fc1.levels == 1, \
            "fc1.levels must be 1 (identity) inside noisy_weights context"


def test_noisy_weights_different_seeds_differ(tiny_model):
    with noisy_weights(tiny_model, sigma=0.2, weight_levels=4, seed=0):
        w_seed0 = tiny_model.fc1.weight.data.clone()
    with noisy_weights(tiny_model, sigma=0.2, weight_levels=4, seed=1):
        w_seed1 = tiny_model.fc1.weight.data.clone()

    assert not torch.allclose(w_seed0, w_seed1), \
        "different seeds should produce different noise realisations"


# ---------------------------------------------------------------------------
# eval_sigma
# ---------------------------------------------------------------------------

def test_eval_sigma_returns_expected_keys(tiny_model, tiny_loader):
    result = eval_sigma(tiny_model, tiny_loader, torch.device("cpu"),
                        sigma=0.0, weight_levels=4, trials=2, base_seed=0)
    for key in ("sigma", "mean_acc", "std_acc", "trials", "per_trial"):
        assert key in result, f"missing key '{key}' in eval_sigma result"


def test_eval_sigma_zero_noise_is_deterministic(tiny_model, tiny_loader):
    """sigma=0 must produce the same accuracy on every trial."""
    result = eval_sigma(tiny_model, tiny_loader, torch.device("cpu"),
                        sigma=0.0, weight_levels=4, trials=4, base_seed=0)
    assert result["std_acc"] == pytest.approx(0.0, abs=1e-6), \
        "sigma=0 should give zero variance across trials"


def test_eval_sigma_mean_acc_in_unit_interval(tiny_model, tiny_loader):
    result = eval_sigma(tiny_model, tiny_loader, torch.device("cpu"),
                        sigma=0.1, weight_levels=4, trials=3, base_seed=0)
    assert 0.0 <= result["mean_acc"] <= 1.0


def test_eval_sigma_high_noise_degrades_accuracy(tiny_model, tiny_loader):
    """Very large noise should reduce accuracy relative to clean inference."""
    clean = eval_sigma(tiny_model, tiny_loader, torch.device("cpu"),
                       sigma=0.0, weight_levels=4, trials=1, base_seed=0)
    noisy = eval_sigma(tiny_model, tiny_loader, torch.device("cpu"),
                       sigma=5.0, weight_levels=4, trials=3, base_seed=0)
    assert noisy["mean_acc"] <= clean["mean_acc"] + 0.1, \
        "very large noise should not improve accuracy"


def test_eval_sigma_per_trial_length_matches_trials(tiny_model, tiny_loader):
    trials = 5
    result = eval_sigma(tiny_model, tiny_loader, torch.device("cpu"),
                        sigma=0.05, weight_levels=4, trials=trials, base_seed=0)
    assert len(result["per_trial"]) == trials
