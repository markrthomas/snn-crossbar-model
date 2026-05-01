"""Tests for AsicFixedPointSpec and CrossbarConfig consistency."""

from __future__ import annotations

import pytest

from src.asic_spec import AsicFixedPointSpec, default_asic_bundle
from src.crossbar_snn import CrossbarConfig


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
