"""Tests for AsicFixedPointSpec and CrossbarConfig consistency."""

from __future__ import annotations

import pytest

from src.asic_spec import AsicFixedPointSpec, default_asic_bundle
from src.crossbar_snn import BYPASS_QUANTIZATION, CrossbarConfig, CrossbarSNN


def test_beta_float_property():
    spec = AsicFixedPointSpec()
    assert spec.beta_float == pytest.approx(983 / 1024)


def test_default_beta_matches_asic_spec():
    """CrossbarConfig default beta must equal AsicFixedPointSpec.beta_float.

    Training and hardware must use the same decay constant; if one is changed
    the other must be updated too.
    """
    cfg = CrossbarConfig()
    spec = AsicFixedPointSpec()
    assert cfg.beta == pytest.approx(spec.beta_float, abs=1e-9)


def test_tile_count_matches_memory_map():
    """tile_count_total in default_asic_bundle must equal the sum of per-layer tile counts.

    crossbar_tile_count() pools cells across layers (underestimates because
    partial-edge tiles in each layer still consume a full crossbar); the memory
    map tiling loop is authoritative.
    """
    bundle = default_asic_bundle(input_dim=784, hidden_dim=256, output_dim=10)
    actual = bundle["memory_map"]["w1"]["tile_count"] + bundle["memory_map"]["w2"]["tile_count"]
    assert bundle["crossbar"]["tile_count_total"] == actual


def test_tile_utilization_denominator_uses_actual_tiles():
    bundle = default_asic_bundle(input_dim=784, hidden_dim=256, output_dim=10)
    xbar_cells = 128 * 128
    actual_tiles = bundle["crossbar"]["tile_count_total"]
    total_cells = bundle["crossbar"]["total_cells"]
    expected_util = total_cells / (actual_tiles * xbar_cells)
    assert bundle["crossbar"]["tile_utilization_total"] == pytest.approx(expected_util)


# ---------------------------------------------------------------------------
# crossbar_report() — tile accounting delegates to default_asic_bundle
# ---------------------------------------------------------------------------

def test_crossbar_report_tile_count_matches_bundle(tiny_cfg):
    model = CrossbarSNN(tiny_cfg)
    report = model.crossbar_report()
    bundle = default_asic_bundle(
        input_dim=tiny_cfg.input_dim,
        hidden_dim=tiny_cfg.hidden_dim,
        output_dim=tiny_cfg.output_dim,
        crossbar_rows=tiny_cfg.crossbar_rows,
        crossbar_cols=tiny_cfg.crossbar_cols,
    )
    assert report["tile_count"] == bundle["crossbar"]["tile_count_total"]
    assert report["tile_utilization"] == pytest.approx(bundle["crossbar"]["tile_utilization_total"])


def test_crossbar_report_cells_correct(tiny_cfg):
    model = CrossbarSNN(tiny_cfg)
    report = model.crossbar_report()
    assert report["layer1_cells"] == tiny_cfg.input_dim * tiny_cfg.hidden_dim
    assert report["layer2_cells"] == tiny_cfg.hidden_dim * tiny_cfg.output_dim
    assert report["total_cells"] == report["layer1_cells"] + report["layer2_cells"]


# ---------------------------------------------------------------------------
# CrossbarConfig.forward_mode
# ---------------------------------------------------------------------------

def test_forward_mode_default_is_snntorch():
    assert CrossbarConfig().forward_mode == "snntorch"


def test_forward_mode_discrete_accepted():
    cfg = CrossbarConfig(forward_mode="discrete")
    assert cfg.forward_mode == "discrete"


def test_forward_mode_invalid_raises():
    with pytest.raises(ValueError, match="forward_mode"):
        CrossbarConfig(forward_mode="bad")


def test_forward_uses_cfg_mode(tiny_cfg):
    import torch
    tiny_cfg_d = CrossbarConfig(
        input_dim=tiny_cfg.input_dim,
        hidden_dim=tiny_cfg.hidden_dim,
        output_dim=tiny_cfg.output_dim,
        num_steps=tiny_cfg.num_steps,
        weight_levels=tiny_cfg.weight_levels,
        crossbar_rows=tiny_cfg.crossbar_rows,
        crossbar_cols=tiny_cfg.crossbar_cols,
        forward_mode="discrete",
    )
    model = CrossbarSNN(tiny_cfg_d).eval()
    images = torch.rand(2, 1, 28, 28)
    logits, _ = model(images)
    assert logits.shape == (2, tiny_cfg.output_dim)


# ---------------------------------------------------------------------------
# CrossbarConfig.validate_asic_compat
# ---------------------------------------------------------------------------

def test_validate_asic_compat_passes_for_defaults():
    CrossbarConfig().validate_asic_compat(AsicFixedPointSpec())  # must not raise


def test_validate_asic_compat_detects_beta_mismatch():
    cfg = CrossbarConfig(beta=0.95)
    with pytest.raises(ValueError, match="beta mismatch"):
        cfg.validate_asic_compat(AsicFixedPointSpec())


def test_validate_asic_compat_detects_excess_weight_levels():
    cfg = CrossbarConfig(weight_levels=300)  # > 2^8 = 256
    with pytest.raises(ValueError, match="weight_levels"):
        cfg.validate_asic_compat(AsicFixedPointSpec())


def test_validate_asic_compat_max_valid_weight_levels():
    cfg = CrossbarConfig(weight_levels=256)  # exactly 2^8
    cfg.validate_asic_compat(AsicFixedPointSpec())  # must not raise


# ---------------------------------------------------------------------------
# BYPASS_QUANTIZATION constant
# ---------------------------------------------------------------------------

def test_bypass_quantization_value():
    assert BYPASS_QUANTIZATION == 1


def test_bypass_quantization_used_in_eval_noise():
    """noisy_weights() must set levels to BYPASS_QUANTIZATION, not a bare literal."""
    import inspect
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.eval_noise import noisy_weights
    src = inspect.getsource(noisy_weights)
    assert "BYPASS_QUANTIZATION" in src
