#!/usr/bin/env python
"""
Generate training/validation loss curves from training_metrics.csv.

Usage:
    python plot_metrics.py [csv_path] [out_dir]
Defaults read ./training_metrics.csv and write PNGs into the same directory.

CSV schema (from PyTorch Lightning's CSVLogger):
    epoch, fit/loss, fit/pred_loss, fit/sigreg_loss, step,
    validate/loss_epoch, validate/loss_step,
    validate/pred_loss_epoch, validate/pred_loss_step,
    validate/sigreg_loss_epoch, validate/sigreg_loss_step

Each row contains EITHER a training-step batch OR a validation entry, not both.
"""
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd


def load(csv_path: Path):
    df = pd.read_csv(csv_path)
    train = df.dropna(subset=["fit/loss"]).copy()
    train["step"] = train["step"].astype(int)
    val = df.dropna(subset=["validate/loss_epoch"]).copy()
    val["step"] = val["step"].astype(int)
    val["epoch"] = val["epoch"].astype(int)
    return train, val


def style_axis(ax, title, ylabel, xlabel="step"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def smooth(y, k=21):
    """Centered moving average; k must be odd. Used to overlay a smoothed curve."""
    s = pd.Series(y).rolling(k, center=True, min_periods=1).mean()
    return s.values


def plot_individual(train, val, out_dir):
    # 1. Total loss
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train["step"], train["fit/loss"], color="#888", alpha=0.4, lw=0.8, label="train (raw, every 50 steps)")
    ax.plot(train["step"], smooth(train["fit/loss"]), color="#1f77b4", lw=2, label="train (smoothed)")
    ax.plot(val["step"], val["validate/loss_epoch"], "o-", color="#d62728", lw=2, ms=7, label="validation (epoch end)")
    style_axis(ax, "Total loss = pred_loss + 0.09 × sigreg_loss", "fit/loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_total.png", dpi=140)
    plt.close(fig)

    # 2. Prediction loss
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train["step"], train["fit/pred_loss"], color="#888", alpha=0.4, lw=0.8, label="train (raw)")
    ax.plot(train["step"], smooth(train["fit/pred_loss"]), color="#2ca02c", lw=2, label="train (smoothed)")
    ax.plot(val["step"], val["validate/pred_loss_epoch"], "o-", color="#d62728", lw=2, ms=7, label="validation")
    style_axis(ax, "Prediction loss (next-latent forecast error)", "fit/pred_loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_pred.png", dpi=140)
    plt.close(fig)

    # 3. SIGReg loss
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train["step"], train["fit/sigreg_loss"], color="#888", alpha=0.4, lw=0.8, label="train (raw)")
    ax.plot(train["step"], smooth(train["fit/sigreg_loss"]), color="#9467bd", lw=2, label="train (smoothed)")
    ax.plot(val["step"], val["validate/sigreg_loss_epoch"], "o-", color="#d62728", lw=2, ms=7, label="validation")
    style_axis(ax, "SIGReg loss (Gaussianity of latent distribution)", "fit/sigreg_loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_sigreg.png", dpi=140)
    plt.close(fig)

    # 4. Same as 3 but log-y to show the spike at epoch 1 cleanly
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train["step"], train["fit/sigreg_loss"], color="#888", alpha=0.4, lw=0.8, label="train (raw)")
    ax.plot(train["step"], smooth(train["fit/sigreg_loss"]), color="#9467bd", lw=2, label="train (smoothed)")
    ax.plot(val["step"], val["validate/sigreg_loss_epoch"], "o-", color="#d62728", lw=2, ms=7, label="validation")
    ax.set_yscale("log")
    style_axis(ax, "SIGReg loss (log scale) — visible end-of-epoch-1 spike", "fit/sigreg_loss (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_sigreg_log.png", dpi=140)
    plt.close(fig)


def plot_dashboard(train, val, out_dir):
    """Single-image 2x2 with all four key panels."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    (ax_total, ax_pred), (ax_sig, ax_sig_log) = axes

    # epoch boundary verticals (from validation step indices)
    ep_steps = val["step"].tolist()

    def vlines(ax):
        for s in ep_steps:
            ax.axvline(s, color="k", lw=0.4, ls="--", alpha=0.25)

    # Total
    ax_total.plot(train["step"], train["fit/loss"], color="#888", alpha=0.4, lw=0.7)
    ax_total.plot(train["step"], smooth(train["fit/loss"]), color="#1f77b4", lw=2, label="train (smoothed)")
    ax_total.plot(val["step"], val["validate/loss_epoch"], "o-", color="#d62728", lw=2, ms=6, label="validation")
    vlines(ax_total)
    style_axis(ax_total, "Total loss", "fit/loss")
    ax_total.legend(fontsize=9)

    # Pred
    ax_pred.plot(train["step"], train["fit/pred_loss"], color="#888", alpha=0.4, lw=0.7)
    ax_pred.plot(train["step"], smooth(train["fit/pred_loss"]), color="#2ca02c", lw=2, label="train (smoothed)")
    ax_pred.plot(val["step"], val["validate/pred_loss_epoch"], "o-", color="#d62728", lw=2, ms=6, label="validation")
    vlines(ax_pred)
    style_axis(ax_pred, "Prediction loss (next-latent MSE)", "fit/pred_loss")
    ax_pred.legend(fontsize=9)

    # SIGReg linear
    ax_sig.plot(train["step"], train["fit/sigreg_loss"], color="#888", alpha=0.4, lw=0.7)
    ax_sig.plot(train["step"], smooth(train["fit/sigreg_loss"]), color="#9467bd", lw=2, label="train (smoothed)")
    ax_sig.plot(val["step"], val["validate/sigreg_loss_epoch"], "o-", color="#d62728", lw=2, ms=6, label="validation")
    vlines(ax_sig)
    style_axis(ax_sig, "SIGReg loss (linear)", "fit/sigreg_loss")
    ax_sig.legend(fontsize=9)

    # SIGReg log
    ax_sig_log.plot(train["step"], train["fit/sigreg_loss"], color="#888", alpha=0.4, lw=0.7)
    ax_sig_log.plot(train["step"], smooth(train["fit/sigreg_loss"]), color="#9467bd", lw=2, label="train (smoothed)")
    ax_sig_log.plot(val["step"], val["validate/sigreg_loss_epoch"], "o-", color="#d62728", lw=2, ms=6, label="validation")
    vlines(ax_sig_log)
    ax_sig_log.set_yscale("log")
    style_axis(ax_sig_log, "SIGReg loss (log)", "fit/sigreg_loss (log)")
    ax_sig_log.legend(fontsize=9)

    fig.suptitle("LeWM TwoRoom training — 8 epochs, batch=128 bf16, RTX 4090 (vertical lines = epoch boundaries)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "loss_dashboard.png", dpi=140)
    plt.close(fig)


def plot_per_epoch_bars(val, out_dir):
    """End-of-epoch validation bars — at-a-glance comparison."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    epochs = val["epoch"].tolist()
    width = 0.27
    x = list(range(len(epochs)))
    ax.bar([i - width for i in x], val["validate/loss_epoch"], width, label="val/loss")
    ax.bar(x, val["validate/pred_loss_epoch"], width, label="val/pred_loss")
    ax.bar([i + width for i in x], val["validate/sigreg_loss_epoch"] / 100, width, label="val/sigreg_loss × 0.01")
    ax.set_xticks(x)
    ax.set_xticklabels([f"epoch {e}" for e in epochs])
    ax.set_ylabel("loss value")
    ax.set_yscale("log")
    ax.set_title("End-of-epoch validation losses (sigreg scaled to fit, log y)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "validation_bars.png", dpi=140)
    plt.close(fig)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "training_metrics.csv"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val = load(csv_path)
    print(f"Loaded: {len(train)} train rows, {len(val)} validation rows")
    print(f"Train step range: {train['step'].min()}..{train['step'].max()}")
    print(f"Validation epochs: {val['epoch'].tolist()}")

    plot_individual(train, val, out_dir)
    plot_dashboard(train, val, out_dir)
    plot_per_epoch_bars(val, out_dir)

    print(f"Wrote PNGs to {out_dir}")


if __name__ == "__main__":
    main()
