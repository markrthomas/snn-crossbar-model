from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import snntorch as snn
import torch
import torch.nn as nn

from src.asic_spec import AsicFixedPointSpec, default_asic_bundle

# Sentinel for QuantLinear.levels: bypasses quantisation in _STEQuantize.forward.
# Used by eval_noise.noisy_weights() to hold pre-computed noisy weights in place
# without re-snapping them to the quantisation grid on every forward call.
# Not a valid CrossbarConfig.weight_levels value (validation rejects it).
BYPASS_QUANTIZATION: int = 1


class _STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, levels: int) -> torch.Tensor:
        if levels <= BYPASS_QUANTIZATION:
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
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 levels: int = 32, noise_sigma: float = 0.0):
        super().__init__(in_features, out_features, bias=bias)
        self.levels = levels
        self.noise_sigma = noise_sigma

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        q_weight = quantize_ste(self.weight, self.levels)
        if self.noise_sigma > 0.0 and self.training:
            q_weight = q_weight + self.noise_sigma * torch.randn_like(q_weight)
        return nn.functional.linear(input_tensor, q_weight, self.bias)


@dataclass
class CrossbarConfig:
    input_dim: int = 28 * 28
    hidden_dim: int = 256
    output_dim: int = 10
    num_steps: int = 25
    beta: float = 983 / 1024  # matches AsicFixedPointSpec.beta_num / beta_den (~0.9599)
    weight_levels: int = 32
    crossbar_rows: int = 128
    crossbar_cols: int = 128
    threshold: float = 1.0
    forward_mode: str = "snntorch"  # "snntorch" or "discrete"

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
        if self.forward_mode not in {"snntorch", "discrete"}:
            raise ValueError(
                f"forward_mode must be 'snntorch' or 'discrete', got {self.forward_mode!r}"
            )

    def validate_asic_compat(self, spec: AsicFixedPointSpec) -> None:
        """Raise ValueError if this config is incompatible with the given ASIC spec.

        Checks that the decay constant and weight quantisation level count are
        consistent with what the RTL and fixed-point references will compute.
        """
        if abs(self.beta - spec.beta_float) > 1e-9:
            raise ValueError(
                f"beta mismatch: config has {self.beta!r}, "
                f"spec has {spec.beta_num}/{spec.beta_den}={spec.beta_float!r}"
            )
        max_levels = 2 ** spec.weight_bits
        if self.weight_levels > max_levels:
            raise ValueError(
                f"weight_levels={self.weight_levels} exceeds the "
                f"{2**spec.weight_bits} representable INT{spec.weight_bits} values"
            )


class CrossbarSNN(nn.Module):
    def __init__(self, cfg: CrossbarConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = QuantLinear(cfg.input_dim, cfg.hidden_dim, bias=False, levels=cfg.weight_levels)
        self.fc2 = QuantLinear(cfg.hidden_dim, cfg.output_dim, bias=False, levels=cfg.weight_levels)
        self.lif1 = snn.Leaky(beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.Leaky(beta=cfg.beta, threshold=cfg.threshold)

    def forward_with_spike_sequence(
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

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = images.shape[0]
        x = images.view(batch_size, -1)
        spikes = torch.stack(
            [torch.bernoulli(torch.clamp(x, 0.0, 1.0)) for _ in range(self.cfg.num_steps)],
            dim=0,
        )
        return self.forward_with_spike_sequence(spikes, mode=self.cfg.forward_mode)

    def set_training_noise(self, sigma: float) -> None:
        """Inject Gaussian noise (std=sigma) into quantised weights during training.

        Simulates RRAM device variability during QAT so the model learns to be
        robust to it.  Noise is active only when model.training is True so eval
        accuracy still reflects the clean quantised weights.
        """
        self.fc1.noise_sigma = sigma
        self.fc2.noise_sigma = sigma

    def crossbar_report(self) -> Dict[str, object]:
        """Return tile and utilisation metrics via the canonical default_asic_bundle()."""
        cfg = self.cfg
        bundle = default_asic_bundle(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            crossbar_rows=cfg.crossbar_rows,
            crossbar_cols=cfg.crossbar_cols,
        )
        xbar = bundle["crossbar"]
        return {
            "crossbar_rows": cfg.crossbar_rows,
            "crossbar_cols": cfg.crossbar_cols,
            "weight_levels": cfg.weight_levels,
            "num_steps": cfg.num_steps,
            "layer1_cells": xbar["layer1_cells"],
            "layer2_cells": xbar["layer2_cells"],
            "total_cells": xbar["total_cells"],
            "tile_count": xbar["tile_count_total"],
            "tile_utilization": xbar["tile_utilization_total"],
        }
