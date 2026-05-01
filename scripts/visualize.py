"""Visualisation tool for snn-crossbar-model training artifacts.

Generates PNG plots from training history, model weights, crossbar tile layout,
noise robustness results, and parameter sweep results.  Any figure whose source
data is missing is silently skipped with a note.

Usage
-----
python scripts/visualize.py                          # all defaults
python scripts/visualize.py --out-dir my_plots/      # custom output directory
python scripts/visualize.py --no-show                # headless / CI mode
python scripts/visualize.py --hidden-dim 128 --weight-levels 16  # match a specific checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; safe in headless / CI environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    sys.exit(
        "matplotlib is required for visualisation:\n"
        "  pip install matplotlib\n"
        "(numpy ships with torch and should already be present)"
    )

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.asic_spec import default_asic_bundle
from src.crossbar_snn import CrossbarConfig, CrossbarSNN, quantize_ste


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def _tile_fill_grid(in_features: int, out_features: int,
                    tile_rows: int, tile_cols: int) -> np.ndarray:
    """Return a 2-D array of per-tile fill fractions (0–1)."""
    n_tr = math.ceil(out_features / tile_rows)
    n_tc = math.ceil(in_features / tile_cols)
    grid = np.zeros((n_tr, n_tc))
    for r in range(n_tr):
        for c in range(n_tc):
            used_rows = min((r + 1) * tile_rows, out_features) - r * tile_rows
            used_cols = min((c + 1) * tile_cols, in_features) - c * tile_cols
            grid[r, c] = (used_rows * used_cols) / (tile_rows * tile_cols)
    return grid


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_training(history: Dict[str, List[float]], out_dir: Path, show: bool) -> None:
    """Loss and accuracy curves from history.json."""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Training History", fontsize=13)

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="train loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], marker="o", label="train acc")
    ax_acc.plot(epochs, history["test_acc"],  marker="s", linestyle="--", label="test acc")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    _save(fig, out_dir / "training.png", show)


def plot_weights(model: CrossbarSNN, cfg: CrossbarConfig, out_dir: Path, show: bool) -> None:
    """Raw vs quantised weight histograms for fc1 and fc2."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Weight Distributions  (weight_levels={cfg.weight_levels})", fontsize=13
    )

    step = 2.0 / (cfg.weight_levels - 1)
    q_levels = [-1.0 + i * step for i in range(cfg.weight_levels)]

    for row, (layer, name) in enumerate([(model.fc1, "fc1  (input→hidden)"),
                                          (model.fc2, "fc2  (hidden→output)")]):
        w = layer.weight.detach().cpu().numpy().ravel()
        q = quantize_ste(layer.weight.detach().cpu(), cfg.weight_levels).numpy().ravel()

        ax_raw = axes[row, 0]
        ax_q   = axes[row, 1]

        ax_raw.hist(w, bins=80, color="steelblue", alpha=0.85)
        ax_raw.set_title(f"{name}: raw weights  (n={len(w):,})")
        ax_raw.set_xlabel("Weight value")
        ax_raw.set_ylabel("Count")
        ax_raw.grid(True, alpha=0.3)

        for lv in q_levels:
            ax_q.axvline(lv, color="red", linewidth=0.6, alpha=0.5)
        ax_q.hist(q, bins=cfg.weight_levels * 2, color="coral", alpha=0.85)
        ax_q.set_title(f"{name}: quantised weights")
        ax_q.set_xlabel("Weight value")
        ax_q.set_ylabel("Count")
        ax_q.grid(True, alpha=0.3)
        ax_q.legend(
            handles=[mpatches.Patch(color="red", alpha=0.5, label="quantisation levels")],
            fontsize=8,
        )

    _save(fig, out_dir / "weights.png", show)


def plot_crossbar_tiles(cfg: CrossbarConfig, out_dir: Path, show: bool) -> None:
    """Crossbar tile layout: fill fraction per tile for fc1 and fc2."""
    bundle = default_asic_bundle(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        crossbar_rows=cfg.crossbar_rows,
        crossbar_cols=cfg.crossbar_cols,
    )
    xbar = bundle["crossbar"]

    layers = [
        ("fc1  (input → hidden)", cfg.input_dim,  cfg.hidden_dim),
        ("fc2  (hidden → output)", cfg.hidden_dim, cfg.output_dim),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Crossbar Tile Layout  ({cfg.crossbar_rows}×{cfg.crossbar_cols} cells/tile)\n"
        f"Total tiles: {xbar['tile_count_total']}   "
        f"Overall utilisation: {xbar['tile_utilization_total']:.1%}",
        fontsize=12,
    )

    for ax, (name, in_f, out_f) in zip(axes, layers):
        grid = _tile_fill_grid(in_f, out_f, cfg.crossbar_rows, cfg.crossbar_cols)
        n_tr, n_tc = grid.shape

        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")

        for r in range(n_tr):
            for c in range(n_tc):
                val = grid[r, c]
                ax.text(
                    c, r, f"{val:.0%}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if val > 0.55 else "black",
                )

        ax.set_xticks(range(n_tc))
        ax.set_xticklabels([f"col {i}" for i in range(n_tc)], fontsize=8)
        ax.set_yticks(range(n_tr))
        ax.set_yticklabels([f"row {i}" for i in range(n_tr)], fontsize=8)
        ax.set_xlabel(f"Tile column  ({cfg.crossbar_cols} input neurons each)")
        ax.set_ylabel(f"Tile row  ({cfg.crossbar_rows} output neurons each)")
        layer_tiles = n_tr * n_tc
        layer_util  = (in_f * out_f) / (layer_tiles * cfg.crossbar_rows * cfg.crossbar_cols)
        ax.set_title(
            f"{name}\n"
            f"matrix {in_f}×{out_f}   {n_tr}×{n_tc} = {layer_tiles} tiles   "
            f"utilisation {layer_util:.1%}",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Tile fill fraction")

    _save(fig, out_dir / "tiles.png", show)


def plot_noise(noise_data: Dict[str, Any], out_dir: Path, show: bool) -> None:
    """Accuracy ± std vs noise sigma (absolute and in LSB units)."""
    results  = noise_data["results"]
    sigmas   = [r["sigma"]    for r in results]
    means    = [r["mean_acc"] for r in results]
    stds     = [r["std_acc"]  for r in results]
    lsb_step = noise_data.get("lsb_step", 1.0)
    baseline = means[0]

    lsb_ratios = [s / lsb_step if lsb_step > 0 else 0.0 for s in sigmas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Noise Robustness  (weight_levels={noise_data['weight_levels']}, "
        f"hidden_dim={noise_data['hidden_dim']}, "
        f"trials={noise_data['trials']})",
        fontsize=12,
    )

    for ax, xs, xlabel in [
        (ax1, sigmas,     "Noise σ  (weight-space units)"),
        (ax2, lsb_ratios, "Noise σ / LSB"),
    ]:
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=5,
                    color="steelblue", ecolor="cornflowerblue", linewidth=2)
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
                   label=f"σ=0 baseline  ({baseline:.4f})")
        lo = max(0.0, min(means) - 0.05)
        hi = min(1.0, baseline  + 0.02)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    ax1.set_title("Absolute noise")
    ax2.set_title("Noise in LSB units  (1 LSB = one weight quantisation step)")

    _save(fig, out_dir / "noise.png", show)


def plot_sweep(configs: List[Dict[str, Any]], out_dir: Path, show: bool) -> None:
    """Heatmap + ranked bar chart from sweep_results.json."""
    wl_vals = sorted({c["weight_levels"] for c in configs})
    ns_vals = sorted({c["num_steps"]     for c in configs})
    hd_vals = sorted({c["hidden_dim"]    for c in configs})

    # One heatmap per hidden_dim value; show all in a column
    n_hd = len(hd_vals)
    fig, axes = plt.subplots(n_hd, 2, figsize=(13, 4 * n_hd + 1),
                             squeeze=False)
    fig.suptitle("QAT Sweep: accuracy vs weight_levels × num_steps", fontsize=13)

    for hd_idx, hd in enumerate(hd_vals):
        subset = [c for c in configs if c["hidden_dim"] == hd]

        # ---- Heatmap ----
        grid = np.full((len(ns_vals), len(wl_vals)), float("nan"))
        for c in subset:
            r   = ns_vals.index(c["num_steps"])
            col = wl_vals.index(c["weight_levels"])
            grid[r, col] = c["best_test_acc"]

        ax_heat = axes[hd_idx, 0]
        valid   = grid[~np.isnan(grid)]
        vmin    = float(valid.min()) if valid.size else 0.0
        vmax    = float(valid.max()) if valid.size else 1.0

        im = ax_heat.imshow(grid, cmap="YlGn", vmin=vmin, vmax=vmax,
                            aspect="auto", interpolation="nearest")
        for r in range(len(ns_vals)):
            for c in range(len(wl_vals)):
                val = grid[r, c]
                if not math.isnan(val):
                    color = "white" if val > (vmin + vmax) / 2 + (vmax - vmin) * 0.2 else "black"
                    ax_heat.text(c, r, f"{val:.3f}", ha="center", va="center",
                                 fontsize=9, color=color)

        ax_heat.set_xticks(range(len(wl_vals)))
        ax_heat.set_xticklabels([str(w) for w in wl_vals])
        ax_heat.set_yticks(range(len(ns_vals)))
        ax_heat.set_yticklabels([str(s) for s in ns_vals])
        ax_heat.set_xlabel("Weight levels")
        ax_heat.set_ylabel("Num steps")
        ax_heat.set_title(f"hidden_dim={hd}  (best test accuracy)")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        # ---- Ranked bar ----
        ax_bar  = axes[hd_idx, 1]
        ranked  = sorted(subset, key=lambda c: c["best_test_acc"], reverse=True)[:8]
        labels  = [
            f"wl={c['weight_levels']}  ns={c['num_steps']}\n"
            f"tiles={c['tile_count']}  bits={c['bits_per_weight']}"
            for c in ranked
        ]
        accs    = [c["best_test_acc"] for c in ranked]
        colors  = ["steelblue" if i == 0 else "cornflowerblue" for i in range(len(ranked))]
        ax_bar.barh(range(len(ranked)), accs, color=colors, alpha=0.85)
        ax_bar.set_yticks(range(len(ranked)))
        ax_bar.set_yticklabels(labels, fontsize=8)
        ax_bar.set_xlabel("Best test accuracy")
        lo = max(0.0, min(accs) - 0.03) if accs else 0.0
        ax_bar.set_xlim(lo, 1.0)
        ax_bar.invert_yaxis()
        ax_bar.set_title(f"hidden_dim={hd}  top configs")
        ax_bar.grid(True, axis="x", alpha=0.3)
        for i, acc in enumerate(accs):
            ax_bar.text(acc + 0.001, i, f"{acc:.4f}", va="center", fontsize=8)

    _save(fig, out_dir / "sweep.png", show)


def plot_epoch_curves(configs: List[Dict[str, Any]], out_dir: Path, show: bool) -> None:
    """Per-config accuracy curves across epochs (one subplot per hidden_dim)."""
    hd_vals = sorted({c["hidden_dim"] for c in configs})
    n_hd    = len(hd_vals)

    fig, axes = plt.subplots(1, n_hd, figsize=(6 * n_hd, 4), squeeze=False)
    fig.suptitle("Test accuracy vs epoch — all sweep configs", fontsize=12)

    cmap = plt.get_cmap("tab20")

    for idx, hd in enumerate(hd_vals):
        ax = axes[0, idx]
        subset = [c for c in configs if c["hidden_dim"] == hd]
        for ci, c in enumerate(sorted(subset, key=lambda x: x["best_test_acc"], reverse=True)):
            epochs = [e["epoch"]    for e in c["epochs"]]
            accs   = [e["test_acc"] for e in c["epochs"]]
            label  = f"wl={c['weight_levels']} ns={c['num_steps']}"
            ax.plot(epochs, accs, marker=".", linewidth=1.5,
                    color=cmap(ci / max(len(subset), 1)), label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"hidden_dim={hd}")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)

    _save(fig, out_dir / "sweep_curves.png", show)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise snn-crossbar-model training artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",    default="./artifacts/best_model.pt",
                   help="Path to trained model checkpoint")
    p.add_argument("--artifacts-dir", default="./artifacts",
                   help="Root directory containing history.json, noise_results.json, sweep/")
    p.add_argument("--out-dir",       default="./artifacts/plots",
                   help="Directory to write PNG files into")
    p.add_argument("--hidden-dim",    type=int, default=256)
    p.add_argument("--weight-levels", type=int, default=32)
    p.add_argument("--num-steps",     type=int, default=25)
    p.add_argument("--crossbar-rows", type=int, default=128)
    p.add_argument("--crossbar-cols", type=int, default=128)
    p.add_argument("--show", action="store_true",
                   help="Call plt.show() after saving each figure (requires a display)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    show = args.show
    if show:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            print("Warning: could not switch to interactive backend; figures will be saved only.")
            show = False

    art = Path(args.artifacts_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = CrossbarConfig(
        hidden_dim=args.hidden_dim,
        num_steps=args.num_steps,
        weight_levels=args.weight_levels,
        crossbar_rows=args.crossbar_rows,
        crossbar_cols=args.crossbar_cols,
    )

    # --- Training curves ---
    history_path = art / "history.json"
    if history_path.exists():
        print("Plotting training history...")
        plot_training(json.loads(history_path.read_text(encoding="utf-8")), out, show)
    else:
        print(f"  skip training.png  ({history_path} not found)")

    # --- Weight distributions (requires checkpoint) ---
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print("Plotting weight distributions...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # Auto-detect hidden_dim from checkpoint so --hidden-dim need not match.
        ckpt_hidden = state_dict["fc1.weight"].shape[0]
        if ckpt_hidden != cfg.hidden_dim:
            print(f"  note: checkpoint hidden_dim={ckpt_hidden} "
                  f"(--hidden-dim={cfg.hidden_dim} overridden for weight plots)")
            w_cfg = CrossbarConfig(
                hidden_dim=ckpt_hidden,
                num_steps=cfg.num_steps,
                weight_levels=cfg.weight_levels,
                crossbar_rows=cfg.crossbar_rows,
                crossbar_cols=cfg.crossbar_cols,
            )
        else:
            w_cfg = cfg
        model = CrossbarSNN(w_cfg)
        model.load_state_dict(state_dict)
        model.eval()
        plot_weights(model, w_cfg, out, show)
    else:
        print(f"  skip weights.png  ({ckpt_path} not found)")

    # --- Crossbar tile layout (always available; derived from config) ---
    print("Plotting crossbar tile layout...")
    plot_crossbar_tiles(cfg, out, show)

    # --- Noise robustness ---
    noise_path = art / "noise_results.json"
    if noise_path.exists():
        print("Plotting noise robustness...")
        plot_noise(json.loads(noise_path.read_text(encoding="utf-8")), out, show)
    else:
        print(f"  skip noise.png  ({noise_path} not found)")

    # --- Sweep ---
    sweep_path = art / "sweep" / "sweep_results.json"
    if sweep_path.exists():
        print("Plotting sweep results...")
        configs = json.loads(sweep_path.read_text(encoding="utf-8"))
        plot_sweep(configs, out, show)
        if any("epochs" in c for c in configs):
            plot_epoch_curves(configs, out, show)
    else:
        print(f"  skip sweep.png  ({sweep_path} not found)")

    print(f"\nPlots written to {out}/")


if __name__ == "__main__":
    main()
