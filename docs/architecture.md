# Architecture

## The motion representation (205-d)

The learning target and model output is the per-frame motion template LivePortrait
extracts from real footage (`src/avagen/features/motion_features.py`):

<img src="assets/figures/fig_motion_dims.png" width="560"/>

| Component | Dims | Meaning |
|---|---|---|
| expression (`exp`) | 63 | facial deformation incl. mouth |
| keypoints (`kp`) | 63 | implicit driving keypoints |
| source keypoints (`x_s`) | 63 | source keypoints |
| rotation (`R`) | 9 | head yaw/pitch/roll (flattened 3×3) |
| translation (`t`) | 3 | head position |
| eye ratio | 2 | eyelid open/close |
| scale | 1 | face size |
| lip ratio | 1 | mouth open/close |

Rotation matrices are **orthonormalized (SVD)** on the way back to a template
(`_project_to_rotation`), and targets are **z-scored** per component before training.

## Flow-matching model (`src/avagen/models/motion_flow.py`)

Predicts the **rectified-flow velocity field** `v(x_t, t, audio)`. Given data motion
`x1` and noise `x0`, `x_t = (1−t)·x0 + t·x1` and the target velocity is the constant
`x1 − x0`.

```
audio (B,T,768) ─ Linear ─┐
motion x_t (B,T,205) ─ Linear ─┼─(+)─► BiGRU (3 layers, hidden 512) ─► LayerNorm+MLP ─► v (B,T,205)
time t (B,) ─ sinusoidal ─ MLP ─┘        (bidirectional, O(T))
```

- **14.69 M params**; hidden 512; 3-layer **bidirectional** GRU (whole clips in one
  pass, no autoregression); sinusoidal time embedding (dim 128); GELU MLP head; dropout 0.1.
- **Classifier-free guidance:** a learned `null_cond` embedding replaces the audio
  projection when audio is dropped (train-time `cfg_p_uncond=0.15`). Sampling:
  `v = v_uncond + w·(v_cond − v_uncond)`.
- **Sampling:** Euler ODE integration of `dx/dt = v` from Gaussian noise over 20 steps.

Why generative? MSE regression to `x1` minimizes average error and collapses to the
**conditional mean** of all plausible motions — a static, mumbling face. Flow
matching learns the *distribution*, so a sample is one full-amplitude, natural motion.

## Other models (implemented)

`motion_gru` (MSE regression baseline, 1.30 M), plus `motion_transformer`,
`motion_tcn`, `motion_cvae` are implemented as alternative temporal backbones; the
GRU and flow variants were the ones trained and evaluated here.

## SyncNet evaluator (`src/avagen/models/sync_net.py`)

A Wav2Lip-style audio↔motion sync discriminator: dual bidirectional-GRU encoders
(audio and motion) → mean-pool → L2-normalized 256-d embeddings; trained with a
symmetric **CLIP-style in-batch contrastive** loss over ~1 s windows. At eval,
cosine similarity of aligned windows gives a **sync-confidence** score, with an
offset sweep that should peak at lag 0. 5.05 M params.

## Rendering

The predicted 205-d sequence is unflattened to a LivePortrait motion template
(`motion_template.py`), optionally post-processed (`motion_postprocess.py`:
global `motion_scale`, translation clamp, blink injection), and rendered by the
frozen LivePortrait warp+decode network against a single source portrait.
