# Compute & systems engineering

All compute figures come from Slurm `sacct`
([`assets/experiment_logs/sacct_compute.raw`](assets/experiment_logs/sacct_compute.raw)).

## Hardware used (Bowdoin HPC, Slurm)

| Stage | GPU | CPUs / Mem | Notes |
|---|---|---|---|
| Model training | **NVIDIA RTX 2080** (11 GB) | 4 / 64 G | GRU, flow, CFG, SyncNet |
| Motion extraction | **NVIDIA RTX Pro 6000** | 4 / 96 G | drives LivePortrait per frame |
| Preprocessing | CPU (`main` partition) | 8 / 32 G | ffmpeg + face crops |
| Rendering | RTX Pro 6000 / A100 | 4 / 32 G | LivePortrait warp+decode |

## Where the time goes

<img src="assets/figures/fig_runtime.png" width="620"/>

- **Motion extraction dominates:** ~**6.4 h** on the Pro 6000 per 16-clip dataset
  (job `64215`, `06:26:42`); the 8-clip Huberman run was `06:16:45`.
- **Generative training is cheap:** GRU ~2 min (60 ep), flow ~12 min (300 ep),
  flow+CFG on 16 clips **21m38s** (300 ep), SyncNet 4m20s (120 ep).
- **Preprocessing:** 17–33 min on CPU nodes.

### Approximate GPU-hours

| Stage | GPU-hours (approx.) |
|---|---|
| Motion extraction (2 datasets, productive) | ~**12.7** |
| Motion extraction (OOM/failed retries) | ~4.7 (wasted) |
| All model training (13 runs combined) | ~**1.3** |
| Rendering (many short jobs) | ~1 |

Extraction is the cost center precisely because the *learning* problem was reduced
to low-dimensional motion — training itself is inexpensive.

## Systems engineering (challenges solved)

- **Slurm round-trip tooling** (`scripts/run_bowdoin_*_roundtrip.sh`): clone ref
  into scratch → submit → poll → fetch artifacts; robust `sacct`-based monitors
  that survive empty-`squeue` blips and VPN drops.
- **GPU-compatibility fix (Exclusive_Process):** the render predicted motion on
  the GPU in the parent process, then spawned LivePortrait as a child needing its
  own CUDA context — denied on Exclusive_Process GPUs (`cudaErrorDevicesUnavailable`).
  Fix: **predict on CPU** so the renderer owns the single context
  (`PREDICT_DEVICE` in `slurm/render_predicted_motion.sbatch`); also route around a
  flaky node and force LivePortrait's onnxruntime to CPU.
- **OOM recovery & template salvage:** a manifest-prep job OOM'd after dumping
  templates; recovered the expensive (6 h) templates from the work dir instead of
  re-extracting.
- **Home-quota/model-cache fix:** wav2vec2 downloads failed on a full home quota;
  redirected `TORCH_HOME`/`HF_HOME` to 32 TB scratch.
- **Alignment bug:** audio↔motion drift from `int`-truncated output fps; fixed to
  use `record.fps` (e.g. 23.976, not 23).

## Storage

Checkpoints (15–176 MB each) and processed datasets (crops + templates +
features) live on HPC scratch and are **not** committed. The final model
checkpoint is 176 MB (includes optimizer state); model-only weights are ~59 MB.
See [reproducibility](reproducibility.md) for regeneration instructions.
