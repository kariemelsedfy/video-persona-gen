# audio-visual-avatar-gen

Personalized audio-visual avatar generation focused on learning audio-driven motion from self-recorded footage and rendering talking-head video with a pretrained LivePortrait-style backbone.

## Session Files

Multiple chats and agents may work in this repository. At the start of each session, read these files in order:

1. `PROJECT_INSTRUCTIONS.md`
2. `PROJECT_CONTEXT.md`
3. `PROJECT_OUTLINE.md`
4. `PROJECT_PROGRESS.md`
5. `SESSION_SUMMARY.md`

Use `CODEX_CONTEXT.md` as the detailed long-form design brief when deeper architectural context is needed.

## Current Scope

This repository is currently focused on Phase 1:

- environment and repo scaffolding
- LivePortrait inference sanity checks
- preprocessing a talking-head clip into audio, frames, face crops, metadata, and manifest entries
- laying the groundwork for later audio-to-motion training

The project is not currently prioritizing UI work, desktop assistants, or agent integrations.

## Target Pipeline

```text
text
  -> voice-cloned audio
  -> audio/prosody features
  -> personalized audio-to-motion model
  -> LivePortrait renderer
  -> generated talking-head video
```

## Repo Layout

```text
configs/        YAML configuration files
scripts/        CLI entrypoints
slurm/          HPC job templates
src/avagen/     Python package
data/           raw, processed, manifests, splits
experiments/    experiment configs, notes, metrics, samples
tests/          smoke and schema tests
```

## First Milestones

1. `source image + driving video -> LivePortrait output video`
2. `raw talking-head video -> audio.wav + face crops + metadata.json + manifest.jsonl`

## Quick Start

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For motion-model training, install the optional training extra in an environment that already has compatible PyTorch wheels:

```bash
pip install -e .[train]
```

Inspect a video:

```bash
python scripts/inspect_video.py path/to/clip.mp4
```

Preprocess a single talking-head clip into audio, crops, metadata, and a manifest:

```bash
python scripts/preprocess_dataset.py \
  --input path/to/clip.mp4 \
  --identity-id self_001
```

Extract a LivePortrait motion template for processed clips:

```bash
python scripts/extract_motion.py \
  --config configs/extract_motion.yaml \
  --manifest-path data/processed/self_001/manifest.jsonl \
  --liveportrait-root external/LivePortrait \
  --python-executable python
```

Assign train/val/test splits for one processed identity:

```bash
python scripts/create_splits.py \
  --identity-id self_001 \
  --processed-root data/processed
```

Extract simple framewise audio features and a prosody summary for processed clips:

```bash
python scripts/extract_audio_features.py \
  --manifest-path data/processed/self_001/manifest.jsonl
```

Extract numeric motion features from `motion_template.pkl` files:

```bash
python scripts/extract_motion_features.py \
  --manifest-path data/processed/self_001/manifest.jsonl
```

Inspect the aligned audio-motion training sequences that come out of the feature pipeline:

```bash
python scripts/inspect_sequence_dataset.py \
  --manifest-path data/processed/self_001/manifest.jsonl
```

Train the first GRU audio-to-motion baseline:

```bash
python scripts/train_motion.py \
  --config configs/train_motion_gru.yaml \
  --manifest-path data/processed/self_001/manifest.jsonl
```

Run the LivePortrait wrapper against an official checkout:

```bash
git clone https://github.com/KlingAIResearch/LivePortrait external/LivePortrait

# Download weights into external/LivePortrait/pretrained_weights
hf download KlingTeam/LivePortrait --local-dir external/LivePortrait/pretrained_weights

python scripts/run_liveportrait_inference.py \
  --config configs/render_liveportrait.yaml \
  --source assets/source_self.png \
  --driving path/to/driving.mp4 \
  --output-dir outputs/samples/liveportrait_demo
```

Render a model-predicted motion template back through LivePortrait:

```bash
python scripts/render_predicted_motion.py \
  --config configs/render_predicted_motion.yaml \
  --checkpoint experiments/exp_004_audio_to_motion_gru/checkpoints/best.pt \
  --manifest-path data/processed/self_001/manifest.jsonl \
  --liveportrait-root external/LivePortrait
```

On July 4, 2026, the upstream `readme.md` command that appends `"README.md" "docs"` after `huggingface-cli download` only fetched those paths on the current CLI. The full-repo `hf download ... --local-dir ...` form above is the working command we validated on Bowdoin HPC.

For Bowdoin HPC specifically, the durable scratch root is now `/mnt/hpc/tmp/<user>/video-persona-gen`. The tracked `slurm/liveportrait_infer_tmp.sbatch` job still stages runtime work under node-local `/tmp`, but it now reuses persistent LivePortrait weights from that scratch root and copies logs plus outputs back there automatically.

To round-trip a real Bowdoin run back onto this machine, use:

```bash
bash scripts/run_bowdoin_liveportrait_roundtrip.sh
```

That wrapper:

- submits `slurm/liveportrait_infer_tmp.sbatch`
- syncs the local `slurm/liveportrait_infer_tmp.sbatch` to the remote Bowdoin repo before submission
- keeps the remote job alive long enough for pickup
- fetches `output/`, `hf.log`, `inference.log`, and `status.env` into `outputs/bowdoin_liveportrait/job-<jobid>/`
- reuses Bowdoin weights from `/mnt/hpc/tmp/<user>/video-persona-gen/liveportrait_weights` instead of redownloading them every run
- signals the remote job to exit after the local download is complete

If you already have a running Bowdoin job in the pickup window, you can attach just the download phase:

```bash
bash scripts/fetch_bowdoin_liveportrait_output.sh \
  --job-id 63748 \
  --local-dir outputs/bowdoin_liveportrait/job-63748
```

## Notes

- LivePortrait itself still expects its own upstream environment and pretrained weights in the external checkout.
- `ffmpeg` and `ffprobe` are required by upstream LivePortrait and by the later preprocessing pipeline.
- The preprocessing path writes clip outputs under `data/processed/<identity_id>/<clip_id>/` with `audio.wav`, `face_crops/`, `frame_metadata.json`, `metadata.json`, and an identity-level `manifest.jsonl`.
- The motion-extraction path writes `motion_template.pkl` back into each processed clip directory and refreshes the manifest’s `motion_template_path` field.
- The dataset-loading path can now read manifest-backed clip records, load clip/frame metadata, load motion templates, and rewrite clip splits through `scripts/create_splits.py`.
- The audio-feature path writes `audio_features.npz` and `prosody_summary.json` back into each processed clip directory and refreshes the manifest with `audio_features_path` plus `prosody_summary_path`.
- The motion-feature path writes `motion_features.npz` and `motion_summary.json` back into each processed clip directory and refreshes the manifest with `motion_features_path` plus `motion_summary_path`.
- The aligned sequence dataset path can now interpolate audio features onto motion-frame timestamps, inspect those sequences from the CLI, and feed the first GRU baseline trainer.
- The predicted-motion render path can now take a trained checkpoint, emit `predicted_motion_template.pkl`, and feed that template directly back into upstream LivePortrait for an actual rendered output video.
- The LivePortrait integration is intended to remain a thin wrapper around an external checkout instead of vendoring that project into this repository.
