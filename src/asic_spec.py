"""Default ASIC-oriented fixed-point and crossbar tiling parameters.

These values are chosen for a first digital ASIC bring-up path:
- INT8 weights (good density vs accuracy for early MNIST experiments)
- Q10.10 style beta using a /1024 denominator (cheap shift+add in hardware)
- 128x128 crossbar tiles (common power-of-two tile; easy floorplan mental model)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class AsicFixedPointSpec:
    weight_bits: int = 8
    weight_scale: int = 128  # float weight -> int8: round(q_float * scale), clamp to [-128,127]
    mem_bits: int = 32
    beta_num: int = 983  # ~0.96 * 1024
    beta_den: int = 1024
    threshold: int = 128  # one "unit" of membrane == one weight LSB after scaling

    def threshold_from_scale(self, scale: int) -> int:
        """Threshold tracks the chosen weight quantization scale (LSB)."""
        return int(scale)


def crossbar_tile_count(total_cells: int, rows: int, cols: int) -> int:
    xbar_cells = rows * cols
    return (total_cells + xbar_cells - 1) // xbar_cells


def memory_map_for_fc(
    *,
    name: str,
    in_features: int,
    out_features: int,
    tile_rows: int,
    tile_cols: int,
    base_addr: int,
) -> Dict[str, Any]:
    """Logical SRAM map: row-major full matrix split into row x col tiles.

    Address unit is one weight word (INT8). Physical packing is implementation-defined;
    this map is for verification, software loaders, and floorplanning discussions.
    """
    total = in_features * out_features
    tiles: List[Dict[str, Any]] = []
    addr = base_addr
    for orow in range(0, out_features, tile_rows):
        for icol in range(0, in_features, tile_cols):
            or_hi = min(orow + tile_rows, out_features) - 1
            ic_hi = min(icol + tile_cols, in_features) - 1
            nrow = or_hi - orow + 1
            ncol = ic_hi - icol + 1
            ncells = nrow * ncol
            tiles.append(
                {
                    "name": f"{name}_tile_{len(tiles)}",
                    "out_row0": orow,
                    "out_row1": or_hi,
                    "in_col0": icol,
                    "in_col1": ic_hi,
                    "nrows": nrow,
                    "ncols": ncol,
                    "ncells": ncells,
                    "base_addr": addr,
                    "end_addr_exclusive": addr + ncells,
                }
            )
            addr += ncells

    return {
        "layer": name,
        "in_features": in_features,
        "out_features": out_features,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "total_cells": total,
        "tile_count": len(tiles),
        "tiles": tiles,
        "end_addr_exclusive": addr,
    }


def default_asic_bundle(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    crossbar_rows: int = 128,
    crossbar_cols: int = 128,
    weight_scale: int | None = None,
) -> Dict[str, Any]:
    fp = AsicFixedPointSpec()
    scale = weight_scale if weight_scale is not None else fp.weight_scale
    threshold = fp.threshold_from_scale(scale)

    layer1_cells = input_dim * hidden_dim
    layer2_cells = hidden_dim * output_dim
    total_cells = layer1_cells + layer2_cells
    xbar_cells = crossbar_rows * crossbar_cols

    map_w1 = memory_map_for_fc(
        name="w1",
        in_features=input_dim,
        out_features=hidden_dim,
        tile_rows=crossbar_rows,
        tile_cols=crossbar_cols,
        base_addr=0,
    )
    map_w2 = memory_map_for_fc(
        name="w2",
        in_features=hidden_dim,
        out_features=output_dim,
        tile_rows=crossbar_rows,
        tile_cols=crossbar_cols,
        base_addr=map_w1["end_addr_exclusive"],
    )

    return {
        "fixed_point": {
            "weight_bits": fp.weight_bits,
            "weight_scale": scale,
            "mem_bits": fp.mem_bits,
            "beta_rational": [fp.beta_num, fp.beta_den],
            "beta_float_approx": fp.beta_num / fp.beta_den,
            "threshold": threshold,
            "rounding": "torch.round (same as numpy.round) on float weights before int cast",
            "clamps": {"weight_int_min": -(2 ** (fp.weight_bits - 1)), "weight_int_max": 2 ** (fp.weight_bits - 1) - 1},
        },
        "crossbar": {
            "rows": crossbar_rows,
            "cols": crossbar_cols,
            "cells_per_tile": xbar_cells,
            "layer1_cells": layer1_cells,
            "layer2_cells": layer2_cells,
            "total_cells": total_cells,
            "tile_count_total": crossbar_tile_count(total_cells, crossbar_rows, crossbar_cols),
            "tile_utilization_total": total_cells / (crossbar_tile_count(total_cells, crossbar_rows, crossbar_cols) * xbar_cells),
        },
        "memory_map": {"w1": map_w1, "w2": map_w2},
    }
