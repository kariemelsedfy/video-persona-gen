# Figures

All figures are generated from **real project data** by
[`scripts/make_portfolio_figures.py`](../../../scripts/make_portfolio_figures.py).
Re-run it after new experiments:

```bash
python3 scripts/make_portfolio_figures.py
```

| File | Question it answers | Data source |
|------|--------------------|-------------|
| `fig_sync_ladder.png` | How much does generative motion beat the regression baseline on lip-sync? | SyncNet `eval_sync_confidence.py` measurements |
| `fig_cfg_sweep.png` | Does classifier-free guidance weight improve sync? | CFG sweep on the 16-clip model, SyncNet-scored |
| `fig_flow_training.png` | Do more data + CFG lower held-out loss? | `train_history.jsonl` (3 flow runs) |
| `fig_gru_ablation.png` | Which audio encoder gives the lowest motion-regression loss? | `train_history.jsonl` (4 GRU runs, HDTF identity) |
| `fig_runtime.png` | Where does wall-clock time go? | Slurm `sacct` elapsed times |
| `fig_motion_dims.png` | What is the 205-dim motion vector made of? | `src/avagen/features/motion_features.py` |

Raw inputs are preserved under [`../experiment_logs/`](../experiment_logs/):
`all_histories.raw` (per-experiment `train_history.jsonl` dumps) and
`sacct_compute.raw` (Slurm accounting). No numbers are hand-entered except the
SyncNet sync-confidence measurements, which are annotated with provenance in the
figure script.
