# Limitations & responsible use

## Technical limitations

- **Person-specific.** Each model is trained on ~1 h of a single speaker and does
  **not** generalize to new identities. A new person requires a new dataset + training run.
- **Same-speaker demo.** The README demo clip's timeline was part of training
  (windowed within-clip split), so it shows **reconstruction quality**, not held-out
  or out-of-distribution generalization. OOD audio and cross-identity transfer are untested.
- **Residual jitter / drift.** Generated motion can show frame-to-frame jitter and
  slow head-pose drift; mitigated but not eliminated by post-processing
  (`motion_scale`, translation clamp, smoothing).
- **Weak eye/blink dynamics.** Eyelid motion is limited; blinks are partly injected
  procedurally rather than fully learned.
- **No explicit emotion control.** Expression follows audio only; no controllable affect.
- **Renderer-bound.** Visual quality is capped by frozen LivePortrait — identity
  degradation, artifacts on large pose, and single-portrait constraints are inherited.
- **Metric noise.** The SyncNet evaluator overfits on limited data; use its
  *rankings*, not absolute values, and corroborate with per-component motion error.
- **Data quality dependence.** Needs clean, mostly frontal footage; low face-detection
  clips are filtered.
- **Compute.** Motion extraction is GPU-heavy (~6.4 h / 16 clips), though training is cheap.

## Responsible use

Synthetic talking-head video of a real person is a **misuse risk** (impersonation,
non-consensual media, disinformation). This project is a personal research /
portfolio artifact.

- Use **only** with identities and footage you are **authorized** to use.
- Do not present generated video as authentic recordings of a person.
- Source data here (podcast / HDTF) is third-party and is **not** redistributed.
- No credentials, private footage, or datasets are committed to this repository.

See also `ETHICS.md` and `DATASET_CARD.md`.
