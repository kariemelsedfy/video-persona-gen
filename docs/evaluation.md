# Evaluation

Evaluation modules live in `src/avagen/evaluation/`
(`sync_metrics`, `motion_metrics`, `identity_metrics`, `quality_metrics`,
`system_metrics`). The headline evaluator is the learned SyncNet metric.

## Why not just MSE?

Motion MSE is misleading here: the near-static MSE-GRU baseline has *low* per-frame
error yet produces a frozen face. A model can win on MSE while failing the actual
task (talking). So the primary metric is a **learned lip-sync score**.

## Sync-confidence (primary)

`scripts/eval_sync_confidence.py` scores audio↔motion agreement with the trained
SyncNet: mean cosine similarity of aligned ~1 s windows, plus an **offset sweep**
that should peak at lag 0 for well-synced motion.

| Model | Sync-confidence | Offset peak |
|---|---|---|
| Ground truth (real motion) | 0.77 | — |
| MSE GRU | 0.09 | −7 |
| Flow (8-clip) | 0.43 | −10 |
| Flow + CFG 16-clip, w=2 | 0.54 | **0** |
| Flow + CFG 16-clip, w=3 | 0.56 | **0** |

Guidance not only raises the score but **fixes time alignment** (offset peak −10 → 0).

**Limitations of the metric:** the SyncNet overfits on ~25 min of one speaker
(train 0.02 / val 2.7), so absolute values are noisy and the ground-truth offset
peak sits at the sweep boundary — but the *ranking* GT > CFG > flow > MSE is stable
and the offset-0 result at w≥2 is meaningful. More data sharpens it.

## Per-component motion error (secondary)

`evaluation/motion_metrics.py` + `scripts/evaluate_motion.py` compute MSE/MAE over
the 205-d vector and **broken down by group** (expression, rotation, translation,
lip, eye), plus temporal velocity/smoothness. Useful for diagnosing *which* motion
channel is off (e.g. translation over-shoot), complementing the sync score.

## Evaluation regimes (be precise about what's tested)

| Regime | Tested here? |
|---|---|
| Training-set reconstruction (same clip timeline) | ✅ (the demo clip) |
| Validation windows (held-out timeline, same clips) | ✅ (val loss) |
| Held-out audio, same speaker | partial (within-clip test region) |
| Out-of-distribution audio | ✗ |
| New identity (generalization) | ✗ — person-specific by design |

The rendered demo is **same-speaker reconstruction**; no cross-identity or OOD
generalization is claimed. See [limitations](limitations.md).

## Reproduce

```bash
python scripts/eval_sync_confidence.py \
  --checkpoint experiments/huberman_syncnet-*/checkpoints/best.pt \
  --manifest-path data/processed/huberman/manifest.jsonl \
  --clip-id huberman_session0_000 \
  --predicted-features .../predicted_motion_features.npz
```
