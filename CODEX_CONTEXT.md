# CODEX_CONTEXT.md

This is the detailed long-form design brief for the project.

For session handoff and multi-agent coordination, also read:

- `PROJECT_INSTRUCTIONS.md`
- `PROJECT_CONTEXT.md`
- `PROJECT_OUTLINE.md`
- `PROJECT_PROGRESS.md`
- `SESSION_SUMMARY.md`

## Project Name

`audio-visual-avatar-gen`

## Project Summary

This project is a deep learning research/engineering project for personalized audio-visual avatar generation.

The final goal is to generate a talking-head video of the user speaking arbitrary text/audio, matching:

- accurate lip sync
- the user's own voice
- the user's head motion
- the user's facial expressions
- the user's speaking rhythm, pauses, delivery style, and mannerisms

The current focus is Phase 1 only: the avatar generation and training pipeline.

Do not focus yet on:

- Claude integration
- Jarvis desktop UI
- screen overlay
- agent tools
- real-time assistant behavior

Those may come later. The current priority is the machine learning system.

## Core Goal

Build a system that can eventually perform:

```text
text
  -> voice-cloned audio
  -> personalized audio/prosody-to-motion model
  -> LivePortrait-style renderer
  -> generated talking-head video
```

The main research question is:

> Can we learn a personalized audio-driven motion model that captures a person's speaking mannerisms, expressions, blinks, and head motion, then use a pretrained portrait animation renderer to generate realistic talking-head video?

## Important Architecture Decision

The project should use LivePortrait as the renderer, not as the entire audio-to-video model.

LivePortrait is useful for:

```text
source image/video + driving motion
  -> animated portrait video
```

But it does not directly solve:

```text
text/audio
  -> personalized head motion/expression/mannerisms
```

Therefore, the main learnable component of this project should be:

```text
audio/prosody/text features
  -> motion representation
```

where motion representation may include:

- head pose
- expression coefficients
- blink/eye features
- mouth/motion controls
- LivePortrait-compatible motion templates or implicit keypoints

Then LivePortrait renders the final video.

## Current Preferred Architecture

```text
Text Input
   ↓
Voice-Cloning TTS
   ↓
Audio / Prosody Feature Extraction
   ↓
Personalized Audio-to-Motion Model
   ↓
LivePortrait Renderer
   ↓
Generated Talking-Head Video
```

The first model to train should not be a full video generator. It should be an audio-to-motion model.

## What Not To Do Initially

Do not start with:

- deep fine-tuning all of LivePortrait
- training a full video generation model from scratch
- building the Jarvis UI
- building Claude/MCP integration
- training on a celebrity as the main path
- relying on a static photo plus generic lip sync as the final approach

Deep LivePortrait fine-tuning is optional later, after the motion model works.

## Subject / Identity Plan

The final personalized model should be trained on footage of the user themselves.

Reason:

- consent is clear
- data can be controlled
- quality can be improved by recording more footage
- validation clips can be held out intentionally
- the project avoids celebrity impersonation issues

Celebrity footage may be used only as an optional noisy-data stress test, not as the main path.

Recommended data path:

```text
clean public benchmark data
  -> LivePortrait/pipeline sanity checks
  -> tiny self-recorded dataset
  -> full self-recorded dataset
  -> personalized motion model
```

## Hardware

Available hardware:

- Local fallback: RTX 3080, 10GB VRAM
- Main training target: Bowdoin HPC Slurm cluster
- Target nodes: `moose68` / `moose69`
- Target GPU: RTX PRO 6000 Blackwell, 96GB VRAM
- Partition: `gpu`
- Slurm should be used for long jobs
- Use `sbatch` for serious training runs
- Use interactive jobs only for short sanity checks

Training should support:

- mixed precision: fp16 or bf16
- checkpointing every 30-60 minutes
- resume from checkpoint
- logging to files
- reproducible configs

## Repository Design Principles

This should be a research-quality repo.

Every experiment should be documented and reproducible.

Use the pattern:

```text
hypothesis
  -> config
  -> run
  -> metrics
  -> samples
  -> conclusion
  -> next experiment
```

Prefer:

- readable code
- modular scripts
- YAML configs
- typed function signatures where helpful
- simple but explicit structure
- command-line runnable scripts
- metrics saved as JSON
- experiment notes saved as Markdown

Avoid:

- hidden notebook-only workflows
- hardcoded paths
- untracked hyperparameters
- undocumented output files
- training runs with no metrics

## Proposed Repository Structure

```text
audio-visual-avatar-gen/
  README.md
  CODEX_CONTEXT.md
  PROJECT_CONTEXT.md
  DATASET_CARD.md
  MODEL_CARD.md
  EXPERIMENTS.md
  BENCHMARKS.md
  ETHICS.md
  requirements.txt
  pyproject.toml
  .gitignore

  configs/
    preprocess_hdtf.yaml
    preprocess_self.yaml
    extract_motion.yaml
    train_motion_gru.yaml
    train_motion_transformer.yaml
    train_motion_cvae.yaml
    train_voice_clone.yaml
    render_liveportrait.yaml
    evaluate.yaml

  scripts/
    inspect_video.py
    preprocess_dataset.py
    create_manifest.py
    create_splits.py
    extract_motion_templates.py
    run_liveportrait_inference.py
    train_motion.py
    train_voice_clone.py
    synthesize_voice.py
    generate_video.py
    evaluate.py

  slurm/
    sanity_check.sbatch
    preprocess.sbatch
    extract_motion.sbatch
    train_motion.sbatch
    train_voice.sbatch
    evaluate.sbatch

  src/
    avagen/
      data/
        dataset.py
        preprocessing.py
        face_tracking.py
        audio.py
        manifests.py
        splits.py

      features/
        audio_features.py
        prosody.py
        motion_features.py

      models/
        motion_gru.py
        motion_tcn.py
        motion_transformer.py
        motion_cvae.py
        losses.py

      renderers/
        liveportrait_wrapper.py
        motion_template.py
        video_renderer.py

      training/
        train_motion.py
        checkpointing.py
        logging.py
        schedulers.py

      inference/
        generate_motion.py
        generate_video.py
        pipeline.py

      evaluation/
        sync_metrics.py
        identity_metrics.py
        motion_metrics.py
        quality_metrics.py
        system_metrics.py

      utils/
        config.py
        paths.py
        video.py
        audio.py
        seed.py
        device.py

  data/
    raw/
    processed/
    manifests/
    splits/

  experiments/
    exp_000_environment_sanity/
    exp_001_hdtf_pipeline/
    exp_002_liveportrait_baseline/
    exp_003_motion_retrieval_baseline/
    exp_004_audio_to_motion_gru/
    exp_005_self_tiny_dataset/
    exp_006_self_tiny_overfit/
    exp_007_self_full_dataset/
    exp_008_self_gru_motion/
    exp_009_self_transformer_motion/
    exp_010_self_transformer_cvae/
    exp_011_voice_clone/
    exp_012_full_text_to_video_pipeline/

  outputs/
    samples/
    checkpoints/
    logs/
    plots/
```

## Ethical Scope

The intended subject is the user themselves.

Do not build the project around impersonating a celebrity.

If public figures or celebrity footage are ever used, treat it only as private research or noisy-data stress testing. Do not publish impersonation outputs.
