from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import snntorch as snn
import torch
import torch.nn as nn


class _STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, levels: int) -> torch.Tensor:
        if levels <= 1:
            return input_tensor
        step = 2.0 / (levels - 1)
        clipped = torch.clamp(input_tensor, -1.0, 1.0)
        quantized = torch.round((clipped + 1.0) / step) * step - 1.0
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output, None


def quantize_ste(input_tensor: torch.Tensor, levels: int) -> torch.Tensor:
    return _STEQuantize.apply(input_tensor, levels)


class QuantLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, levels: int = 32):
        super().__init__(in_features, out_features, bias=bias)
        self.levels = levels

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        q_weight = quantize_ste(self.weight, self.levels)
        return nn.functional.linear(input_tensor, q_weight, self.bias)


@dataclass
class CrossbarConfig:
    input_dim: int = 28 * 28
    hidden_dim: int = 256
    output_dim: int = 10
    num_steps: int = 25
    beta: float = 0.95
    weight_levels: int = 32
    crossbar_rows: int = 128
    crossbar_cols: int = 128
    threshold: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")
        if self.threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.weight_levels < 2:
            raise ValueError(f"weight_levels must be >= 2, got {self.weight_levels}")
        for name, val in [
            ("input_dim", self.input_dim),
            ("hidden_dim", self.hidden_dim),
            ("output_dim", self.output_dim),
            ("num_steps", self.num_steps),
            ("crossbar_rows", self.crossbar_rows),
            ("crossbar_cols", self.crossbar_cols),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")


class CrossbarSNN(nn.Module):
    def __init__(self, cfg: CrossbarConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = QuantLinear(cfg.input_dim, cfg.hidden_dim, bias=False, levels=cfg.weight_levels)
        self.fc2 = QuantLinear(cfg.hidden_dim, cfg.output_dim, bias=False, levels=cfg.weight_levels)
        self.lif1 = snn.Leaky(beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.Leaky(beta=cfg.beta, threshold=cfg.threshold)

    def _forward_with_spikes(
        self,
        spikes: torch.Tensor,
        mode: str = "snntorch",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if mode not in {"snntorch", "discrete"}:
            raise ValueError(f"Unsupported mode: {mode}")

        num_steps, batch_size, _ = spikes.shape
        mem1 = torch.zeros(batch_size, self.cfg.hidden_dim, device=spikes.device)
        mem2 = torch.zeros(batch_size, self.cfg.output_dim, device=spikes.device)

        spike_count_l1 = 0.0
        spike_count_l2 = 0.0
        spk2_rec = []

        for step in range(num_steps):
            spk_in = spikes[step]
            cur1 = self.fc1(spk_in)
            if mode == "snntorch":
                spk1, mem1 = self.lif1(cur1, mem1)
            else:
                mem1_pre = self.cfg.beta * mem1 + cur1
                spk1 = (mem1_pre >= self.cfg.threshold).to(cur1.dtype)
                mem1 = mem1_pre - spk1 * self.cfg.threshold

            cur2 = self.fc2(spk1)
            if mode == "snntorch":
                spk2, mem2 = self.lif2(cur2, mem2)
            else:
                mem2_pre = self.cfg.beta * mem2 + cur2
                spk2 = (mem2_pre >= self.cfg.threshold).to(cur2.dtype)
                mem2 = mem2_pre - spk2 * self.cfg.threshold

            spike_count_l1 += float(spk1.sum().item())
            spike_count_l2 += float(spk2.sum().item())
            spk2_rec.append(spk2)

        logits = torch.stack(spk2_rec).sum(dim=0)
        stats = {
            "avg_spikes_l1_per_sample": spike_count_l1 / max(batch_size, 1),
            "avg_spikes_l2_per_sample": spike_count_l2 / max(batch_size, 1),
        }
        return logits, stats

    def forward_with_spike_sequence(
        self,
        spikes: torch.Tensor,
        mode: str = "snntorch",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self._forward_with_spikes(spikes, mode=mode)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = images.shape[0]
        x = images.view(batch_size, -1)
        spikes = torch.stack(
            [torch.bernoulli(torch.clamp(x, 0.0, 1.0)) for _ in range(self.cfg.num_steps)],
            dim=0,
        )
        return self._forward_with_spikes(spikes, mode="snntorch")

    def crossbar_report(self) -> Dict[str, float]:
        cfg = self.cfg
        xbar_cells = cfg.crossbar_rows * cfg.crossbar_cols
        layer_1_cells = cfg.input_dim * cfg.hidden_dim
        layer_2_cells = cfg.hidden_dim * cfg.output_dim
        total_cells = layer_1_cells + layer_2_cells
        num_tiles = (total_cells + xbar_cells - 1) // xbar_cells
        utilization = total_cells / (num_tiles * xbar_cells)

        return {
            "crossbar_rows": cfg.crossbar_rows,
            "crossbar_cols": cfg.crossbar_cols,
            "weight_levels": cfg.weight_levels,
            "num_steps": cfg.num_steps,
            "layer1_cells": layer_1_cells,
            "layer2_cells": layer_2_cells,
            "total_cells": total_cells,
            "tile_count": num_tiles,
            "tile_utilization": utilization,
        }
