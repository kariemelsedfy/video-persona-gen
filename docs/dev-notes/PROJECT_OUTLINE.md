# Project Outline

This is the short, stable project outline. It should change rarely.

For the full detailed brief, see `CODEX_CONTEXT.md`.

## Project Goal

Build a personalized audio-visual avatar generation system that learns a person's speaking motion and renders a talking-head video using a pretrained LivePortrait-style backbone.

## Current Scope

The project is currently focused on Phase 1:

- repository and experiment scaffolding
- LivePortrait inference sanity checks
- preprocessing pipeline for talking-head clips
- later audio-to-motion model training

Do not prioritize UI work, assistant integrations, or real-time desktop features yet.

## Core Architecture

```text
text
  -> voice-cloned audio
  -> audio/prosody features
  -> personalized audio-to-motion model
  -> LivePortrait renderer
  -> generated talking-head video
```

The first learnable component should be the audio-to-motion model, not a full video generator.

## Key Decisions

- Use LivePortrait as a renderer, not as the full audio-to-video solution.
- Train primarily on self-recorded footage for consent, control, and iteration speed.
- Validate the pipeline on public talking-head data before scaling self data collection.
- Keep the repository reproducible, experiment-driven, and Slurm-friendly.

## Immediate Milestones

1. `source image + driving video -> LivePortrait output video`
2. `raw talking-head video -> processed audio, face crops, metadata, manifest`
3. `public benchmark subset -> pipeline sanity metrics`
4. `audio features -> motion baseline -> rendered validation output`

## Non-Goals For Now

- deep LivePortrait fine-tuning
- full video generation from scratch
- celebrity-first training
- Jarvis or desktop assistant UI
- Claude or MCP product integration
