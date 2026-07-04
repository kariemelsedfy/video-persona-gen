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

Inspect a video stub:

```bash
python scripts/inspect_video.py path/to/clip.mp4
```

Preprocess a single clip stub:

```bash
python scripts/preprocess_dataset.py \
  --input path/to/clip.mp4 \
  --identity-id self_001
```

Run the LivePortrait wrapper against an official checkout:

```bash
git clone https://github.com/KlingAIResearch/LivePortrait external/LivePortrait

# Download weights into external/LivePortrait/pretrained_weights
# See the official LivePortrait README for the current recommended command.

python scripts/run_liveportrait_inference.py \
  --config configs/render_liveportrait.yaml \
  --source assets/source_self.png \
  --driving path/to/driving.mp4 \
  --output-dir outputs/samples/liveportrait_demo
```

## Notes

- LivePortrait itself still expects its own upstream environment and pretrained weights in the external checkout.
- `ffmpeg` and `ffprobe` are required by upstream LivePortrait and by the later preprocessing pipeline.
- The LivePortrait integration is intended to remain a thin wrapper around an external checkout instead of vendoring that project into this repository.
