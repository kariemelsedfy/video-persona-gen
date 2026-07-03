# Dataset Card

## Intended Datasets

- public talking-head data for preprocessing and renderer sanity checks
- self-recorded footage for the personalized model

## Processed Dataset Layout

```text
data/processed/<identity_id>/
  clips/
    clip_000001/
      audio.wav
      frames/
      face_crops/
      metadata.json
  manifest.jsonl
  dataset_report.json
```

## Required Metadata

- `clip_id`
- `identity_id`
- `audio_path`
- `face_crop_dir`
- `fps`
- `duration_sec`
- `num_frames`
- `face_detection_rate`
- `audio_sample_rate`
- `split`

Landmarks, pose, expression, and motion templates are staged for later extraction passes.

## Data Policy

- keep raw and processed media out of git
- split by clip or session, never random frame leakage
- prefer self footage as the primary personalized dataset
