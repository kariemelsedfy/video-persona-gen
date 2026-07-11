# Data pipeline

From long-form talking-head footage to aligned `(audio_features, motion_targets)`
training windows. All stages are scripted (`scripts/`) and Slurm-wrapped (`slurm/`).

```
raw video ─► preprocess ─► face crops + audio.wav ─► LivePortrait motion extraction ─► motion_template.pkl
                                       │                                                      │
                                       ├─► wav2vec2 features (768-d) ──────────┐              ├─► motion_features.npz (205-d)
                                       └─► mel / prosody features              │              │
                                                                               ▼              ▼
                                                          windowed within-clip split + z-score normalization
                                                                               ▼
                                                       aligned (audio, motion) windows → model
```

## Stages

1. **Preprocess** (`preprocess_dataset.py`, CPU): resample to a fixed fps (20 fps
   here), extract `audio.wav` (16 kHz), detect+crop the face per frame
   (`data/face_tracking.py`, Haar cascade), write `frame_metadata.json`,
   `metadata.json`, and an identity `manifest.jsonl`. Even-dimension crop fix for libx264.
2. **Motion extraction** (`extract_motion.py`, GPU): drive frozen LivePortrait with
   the real face crops to dump the per-frame **motion template** = ground truth.
   This is the expensive stage (~6.4 h / 16 clips, see [compute](compute.md)).
3. **Audio features**: `extract_wav2vec_features.py` (768-d, layer 8, chunked to
   avoid O(T²) attention on long clips, resampled onto the motion grid) and
   `extract_audio_features.py` (mel + prosody).
4. **Motion features** (`extract_motion_features.py`): flatten the template to the
   205-d vector; store per-component stats.
5. **Windowing + splits** (`data/windowing.py`, `create_splits.py`): a **within-clip
   temporal split** carves each clip's timeline into train/val/test regions (so
   train and eval never overlap in time), then slices fixed-length windows.
6. **Alignment + normalization** (`data/dataset.py`): audio features are
   interpolated onto motion-frame timestamps using the clip's **true fps**
   (`record.fps`, e.g. 23.976 — not the `int`-truncated output fps, which caused
   drift); motion targets are z-scored per component.

## Dataset statistics

| Identity | Source | Clips | fps | ~Frames | Notes |
|---|---|---|---|---|---|
| **Andrew Huberman** | 16 YouTube podcast segments | 16 | 20 | ~76,800 (16 × 4,800) | primary; ~64 min |
| **HDTF "CMR"** | HDTF talking-head | 8 | 20 | — | secondary / ablation |

Window counts per experiment are in [experiments.md](experiments.md) (e.g. the
final 16-clip model: 1216 train / 144 val windows).

## Storage format

Per clip under `data/processed/<identity>/<clip_id>/`:
`audio.wav`, `face_crops/`, `frame_metadata.json`, `metadata.json`,
`motion_template.pkl`, `motion_features.npz`, `audio_features.npz`
(mel/prosody + `wav2vec` key), plus an identity-level `manifest.jsonl` and
`dataset_report.json`.

## Data & licensing

Source footage is third-party (podcast / HDTF) and is **not committed** (raw and
processed data are git-ignored). To build an equivalent dataset, run the pipeline
on **footage you are authorized to use** — see [reproducibility](reproducibility.md).
