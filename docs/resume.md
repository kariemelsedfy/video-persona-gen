# Résumé bullets

Drawn strictly from measured project facts (see [experiments](experiments.md),
[compute](compute.md)). Numbers: ~64 min / 16-clip single-speaker dataset,
~76,800 motion frames, 205-d motion vector, 768-d wav2vec2 conditioning, 14.7 M-param
flow model, RTX 2080 training / RTX Pro 6000 extraction, ~14 GPU-hours total,
sync-confidence 0.09 → 0.56 (GT 0.77).

## Short (pick 3–4)

- Built an **audio-driven talking-head generator** that learns a person's facial
  **motion coefficients** (205-d/frame) from ~1 h of footage and renders them with a
  frozen portrait-animation model — reducing audio→video to low-dimensional motion regression.
- Improved lip-sync **~6×** over an MSE regression baseline (sync-confidence 0.09 →
  0.56 vs. 0.77 ground truth) by reframing motion prediction as **rectified
  flow-matching with classifier-free guidance**, fixing the mean-collapse "frozen-face" failure.
- Implemented a **learned SyncNet audio↔motion sync metric** (CLIP-style contrastive)
  to evaluate lip-sync objectively where MSE is misleading.
- Engineered an end-to-end **audiovisual data pipeline** (face tracking, LivePortrait
  motion extraction, wav2vec2 features, temporal alignment/windowing) and ran all
  training on a **Slurm HPC** with custom submit/monitor/fetch tooling.

## Detailed

- **Generative motion modeling:** designed and trained a 14.7 M-param rectified
  flow-matching model (3-layer BiGRU velocity field, sinusoidal time embedding,
  20-step Euler ODE sampling) that maps 768-d wav2vec2 features to 205-d LivePortrait
  motion vectors; added classifier-free guidance (audio dropout + guided sampling),
  raising sync-confidence 0.43 → 0.56 and correcting audio-motion time alignment.
- **Data engineering:** turned 16 long-form podcast segments (~64 min, ~76,800
  frames) into aligned training windows via face-crop extraction, GPU motion-template
  extraction, chunked wav2vec2, within-clip temporal splitting, and per-component
  z-score normalization; fixed an fps-truncation audio-drift bug.
- **HPC/systems:** ran preprocessing, ~6.4 h/16-clip GPU motion extraction, and
  generative training across RTX 2080 / RTX Pro 6000 nodes via Slurm; diagnosed and
  fixed a LivePortrait render failure on Exclusive_Process GPUs (CPU-side prediction
  so the renderer owns the single CUDA context) and recovered 6 h of templates after an OOM.
- **Evaluation:** implemented per-component motion-error metrics and a trained SyncNet
  lip-sync evaluator; ran ablations over audio encoders (prosody/mel/wav2vec2),
  objective (MSE vs flow), guidance weight, and dataset size.

## One-liner

> Audio-driven talking-head synthesis: learn a person's 205-d facial-motion
> coefficients from speech with flow-matching + CFG, render via frozen LivePortrait;
> **6× lip-sync gain** over the regression baseline, trained end-to-end on Slurm HPC.
