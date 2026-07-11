#!/usr/bin/env python3
"""Generate the portfolio figures for the README / docs from real experiment data.

Every number here is traceable to project evidence:

* Loss curves            -> docs/assets/experiment_logs/all_histories.raw
                            (train_history.jsonl pulled from each HPC experiment dir)
* Runtimes / GPU-hours   -> docs/assets/experiment_logs/sacct_compute.raw (Slurm sacct)
* Sync-confidence numbers -> measured this project with the trained SyncNet
                            (scripts/eval_sync_confidence.py); values recorded in
                            SYNC_LADDER / CFG_SWEEP below with provenance.
* Motion-vector layout   -> src/avagen/features/motion_features.py (205-dim template)

Figures are written to docs/assets/figures/*.png. Re-run after new experiments:

    python3 scripts/make_portfolio_figures.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "docs" / "assets" / "experiment_logs" / "all_histories.raw"
FIGDIR = REPO / "docs" / "assets" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- Okabe-Ito colourblind-safe categorical palette (validated: CVD dE ~37) ----
BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
GREY, INK, MUTED = "#9aa0a6", "#1a1a1a", "#5f6368"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#ececec",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load_histories() -> dict[str, list[dict]]:
    """Parse the @@@EXP=... markered dump into {experiment_name: [row, ...]}."""
    out: dict[str, list[dict]] = {}
    cur = None
    for line in LOGS.read_text().splitlines():
        m = re.match(r"@@@EXP=(\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        line = line.strip()
        if cur and line.startswith("{"):
            try:
                out[cur].append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _val_loss(row: dict) -> float | None:
    if "val_loss" in row and row["val_loss"] is not None:
        return float(row["val_loss"])
    if isinstance(row.get("val"), dict) and row["val"].get("loss") is not None:
        return float(row["val"]["loss"])
    return None


def _train_loss(row: dict) -> float | None:
    if "train_loss" in row and row["train_loss"] is not None:
        return float(row["train_loss"])
    if isinstance(row.get("train"), dict) and row["train"].get("loss") is not None:
        return float(row["train"]["loss"])
    return None


# ============================ 1. Sync-confidence ladder ============================
# Measured with the trained SyncNet on huberman_session0_000 (eval_sync_confidence.py).
# MSE-GRU 0.0925 | flow (8-clip) 0.4340 | flow+CFG w=3 (16-clip) 0.5585 | GT 0.7662
SYNC_LADDER = [
    ("MSE GRU\n(regression baseline)", 0.09, VERM),
    ("Flow matching\n(8 clips)", 0.43, ORANGE),
    ("Flow + CFG\n(16 clips, w=3)", 0.56, BLUE),
    ("Ground truth\n(real motion)", 0.77, GREY),
]


def fig_sync_ladder():
    labels = [l for l, _, _ in SYNC_LADDER]
    vals = [v for _, v, _ in SYNC_LADDER]
    cols = [c for _, _, c in SYNC_LADDER]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cols, height=0.62, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left",
                fontsize=12, fontweight="bold", color=INK)
    ax.axvline(0.77, color=GREY, ls="--", lw=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.86)
    ax.set_xlabel("SyncNet sync-confidence  (higher = tighter audio-lip sync)")
    ax.set_title("Lip-sync quality: generative motion vs. regression baseline")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_sync_ladder.png", bbox_inches="tight")
    plt.close(fig)


# ============================ 2. CFG guidance-weight sweep ============================
# Measured on the 16-clip CFG model (huberman_flowcfg16) with eval_sync_confidence.py.
CFG_SWEEP = [(1.0, 0.418), (1.5, 0.506), (2.0, 0.541), (3.0, 0.559)]
GT_SYNC = 0.766


def fig_cfg_sweep():
    ws = [w for w, _ in CFG_SWEEP]
    ss = [s for _, s in CFG_SWEEP]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.axhline(GT_SYNC, color=GREY, ls="--", lw=1.2, zorder=1)
    ax.text(3.0, GT_SYNC + 0.006, "ground truth 0.77", ha="right", va="bottom",
            color=MUTED, fontsize=10)
    ax.plot(ws, ss, "-o", color=BLUE, lw=2.0, ms=8, zorder=3)
    for w, s in CFG_SWEEP:
        ax.text(w, s - 0.018, f"{s:.2f}", ha="center", va="top", fontsize=10,
                fontweight="bold", color=INK)
    ax.set_xlabel("Classifier-free guidance weight  w")
    ax.set_ylabel("Sync-confidence")
    ax.set_title("CFG guidance tightens lip-sync (16-clip model)")
    ax.set_xticks(ws)
    ax.set_ylim(0.38, 0.80)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_cfg_sweep.png", bbox_inches="tight")
    plt.close(fig)


# ============================ 3. Flow training curves ============================
def fig_flow_training(hist):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    runs = [
        ("huberman_flow_w2v-20260708-173216", "Flow (8 clips)", ORANGE),
        ("huberman_flowcfg-20260709-213712", "Flow + CFG (8 clips)", PURPLE),
        ("huberman_flowcfg16-20260710-161733", "Flow + CFG (16 clips)", BLUE),
    ]
    for key, label, col in runs:
        rows = hist.get(key, [])
        ep = [r["epoch"] for r in rows]
        vl = [_val_loss(r) for r in rows]
        ax.plot(ep, vl, color=col, lw=1.9, label=label)
        if ep:
            ax.text(ep[-1], vl[-1], f" {vl[-1]:.2f}", va="center", ha="left",
                    fontsize=9, color=col, fontweight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation flow-matching loss")
    ax.set_title("More data + CFG lower the held-out flow loss")
    ax.legend(frameon=False, loc="upper right")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_flow_training.png", bbox_inches="tight")
    plt.close(fig)


# ============================ 4. GRU audio-feature ablation (CMR) ============================
# Final validation target_metric (normalized motion MSE + velocity) per audio encoder.
def fig_gru_ablation(hist):
    runs = [
        ("hdtf_cmr_gru_normalized-20260706-153137", "Prosody (5-d)", 5),
        ("hdtf_cmr_gru_mel-20260706-153714", "Log-mel (42-d)", 42),
        ("hdtf_cmr_gru_melbi-20260706-155154", "Log-mel + BiGRU (42-d)", 42),
        ("hdtf_cmr_gru_w2v-20260706-225130", "wav2vec2 (768-d)", 768),
    ]
    labels, best = [], []
    for key, label, _ in runs:
        rows = hist.get(key, [])
        vls = [_val_loss(r) for r in rows if _val_loss(r) is not None]
        if vls:
            labels.append(label)
            best.append(min(vls))
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(labels))
    cols = [GREEN, ORANGE, BLUE, PURPLE][: len(labels)]
    ax.bar(x, best, color=cols, width=0.62, zorder=3)
    for xi, v in zip(x, best):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("best validation loss (lower = better)")
    ax.set_title("Audio-encoder ablation (GRU, HDTF identity)")
    ax.set_ylim(0, max(best) * 1.18)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_gru_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ============================ 5. Runtime by job type ============================
# From sacct_compute.raw (Elapsed). Representative values in minutes.
RUNTIMES = [
    ("GRU train\n(60 ep)", 2.0, GREEN),
    ("SyncNet\n(120 ep)", 4.3, PURPLE),
    ("Flow train\n(300 ep)", 12.3, ORANGE),
    ("Flow+CFG 16-clip\n(300 ep)", 21.6, BLUE),
    ("Motion extraction\n(16 clips)", 386.7, VERM),
]


def fig_runtime():
    labels = [l for l, _, _ in RUNTIMES]
    vals = [v for _, v, _ in RUNTIMES]
    cols = [c for _, _, c in RUNTIMES]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=cols, width=0.6, zorder=3)
    for xi, v in zip(x, vals):
        lab = f"{v:.0f} min" if v >= 10 else f"{v:.1f} min"
        if v > 60:
            lab = f"{v/60:.1f} h"
        ax.text(xi, v * 1.05, lab, ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("wall-clock time (min, log scale)")
    ax.set_title("Per-stage wall-clock time  ·  extraction dominates, training is cheap")
    _despine(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_runtime.png", bbox_inches="tight")
    plt.close(fig)


# ============================ 6. Motion-vector composition ============================
# 205-dim LivePortrait motion template (src/avagen/features/motion_features.py).
MOTION_DIMS = [
    ("expression", 63, BLUE),
    ("keypoints", 63, ORANGE),
    ("source kp", 63, GREEN),
    ("rotation", 9, PURPLE),
    ("translation", 3, VERM),
    ("eye ratio", 2, MUTED),
    ("scale", 1, GREY),
    ("lip ratio", 1, INK),
]


def fig_motion_dims():
    labels = [l for l, _, _ in MOTION_DIMS]
    vals = [v for _, v, _ in MOTION_DIMS]
    cols = [c for _, _, c in MOTION_DIMS]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cols, height=0.66, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.6, yi, str(v), va="center", ha="left", fontsize=10,
                fontweight="bold", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("dimensions")
    ax.set_xlim(0, 70)
    ax.set_title("Per-frame motion vector: 205 dims LivePortrait predicts")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_motion_dims.png", bbox_inches="tight")
    plt.close(fig)


def main():
    hist = load_histories()
    print(f"parsed {len(hist)} experiments from {LOGS.name}")
    fig_sync_ladder()
    fig_cfg_sweep()
    fig_flow_training(hist)
    fig_gru_ablation(hist)
    fig_runtime()
    fig_motion_dims()
    print(f"wrote figures to {FIGDIR}")
    for p in sorted(FIGDIR.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
