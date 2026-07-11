# Training

Configs live in `configs/`; the trainer is `src/avagen/training/train_flow.py`
(flow/CFG), `train_motion.py` (GRU), `train_syncnet.py` (evaluator). Launch via
`scripts/train_*.py` locally or `slurm/train_motion.sbatch` on the cluster.

## Final model setup (`huberman_flowcfg16`)

| | |
|---|---|
| Objective | rectified flow-matching velocity MSE (masked) |
| Optimizer | **AdamW**, lr `2e-4`, weight decay `0` |
| Grad clipping | global-norm `1.0` |
| Epochs | 300 |
| Precision | fp32 |
| Windowing | within-clip temporal split; z-scored motion targets |
| CFG | audio dropout `cfg_p_uncond = 0.15`; learned `null_cond` |
| Sampling (eval) | 20-step Euler ODE; guidance weight `w` |
| Checkpointing | best (by val loss) + last every epoch |
| GPU / runtime | RTX 2080 · 21m38s |

Validation loss is averaged over several random flow-times `t` per batch for a
stable estimate. Best checkpoint selected by held-out flow loss (0.226 @ ep 289).

## Deep-learning techniques used (and where)

| Technique | Where | Why |
|---|---|---|
| **Transfer learning / frozen feature extractor** | `features/wav2vec_features.py` | pretrained wav2vec2 gives phonetic audio features without paired data |
| **Rectified flow matching** | `models/motion_flow.py` | generative motion — avoids MSE mean-collapse |
| **Classifier-free guidance** | `motion_flow.py`, `train_flow.py` | audio dropout + guided sampling → tighter lip-sync |
| **Contrastive learning (CLIP-style)** | `models/sync_net.py` | learn an objective audio↔motion sync metric |
| **Velocity / smoothness loss** | GRU trainer (`reconstruction_loss` + `velocity_loss`) | penalize jitter in the regression baseline |
| **Target normalization** | `training/train_motion.py` `_compute_motion_normalization` | z-score per motion component; denormalized at inference |
| **Rotation orthonormalization (SVD)** | `features/motion_features.py` | keep predicted `R` a valid rotation |
| **Windowed within-clip split** | `data/windowing.py` | no train/eval temporal overlap; more windows from little data |
| **Gradient clipping / weight decay / dropout** | trainers | overfitting control on a small person-specific set |
| **Bidirectional temporal modeling** | `motion_flow.py`, `motion_gru.py` | full-sequence context, O(T) per pass |
| **Deterministic seeding** | `utils/seed.py` | reproducibility |

## Overfitting control

The datasets are small (person-specific, ~1 h). Mitigations: within-clip windowing
to maximize windows, dropout (0.1–0.4 across variants), weight decay/regularized
variants (`gru_w2vreg`), and — most effective — **more data** (8→16 clips lowered
held-out flow loss 0.29 → 0.23). The SyncNet evaluator also overfits on ~25 min of
one speaker (train loss 0.02 vs val 2.7), which is why more data sharpens *both* the
generator and the metric.

## Reproduce

```bash
python scripts/train_flow.py \
  --config configs/train_flow_w2v_cfg.yaml \
  --manifest-path data/processed/<identity>/manifest.jsonl \
  --experiment-dir experiments/<name>
```
On Slurm: `TRAIN_SCRIPT=scripts/train_flow.py CONFIG_PATH=configs/train_flow_w2v_cfg.yaml ... sbatch slurm/train_motion.sbatch`.
