# Project Progress

This is the current working-state tracker. Update it whenever repository state or next actions change.

## Status

- Date: July 3, 2026
- Phase: LivePortrait inference integration in progress
- Branch: `feature/liveportrait-inference-smoke`
- Overall state: the repo now has a real LivePortrait wrapper path, but upstream checkout and pretrained-weight validation on this machine are still pending

## Completed

- Added the top-level repository skeleton for configs, scripts, Slurm, source package modules, experiments, and docs.
- Added documentation scaffolding for ethics, benchmarks, dataset/model cards, and experiments.
- Added placeholder CLI entrypoints and package modules rather than implementing the real pipeline yet.
- Added shared session coordination files for multi-chat and multi-agent handoff.
- Replaced the LivePortrait wrapper stub with real command-building and subprocess execution against an external official checkout.
- Added a smoke-style wrapper test and updated docs/config to expect `external/LivePortrait`.

## Current Reality

- `scripts/run_liveportrait_inference.py` and `src/avagen/renderers/liveportrait_wrapper.py` now implement a real external-checkout wrapper path.
- `scripts/inspect_video.py`, `scripts/preprocess_dataset.py`, and `scripts/create_manifest.py` are still stubs.
- The preprocessing, face tracking, audio extraction, and model-training modules are still placeholders or explicit stubs.
- This shell currently does not have `ffmpeg`, `huggingface-cli`, `git-lfs`, or `pytest`, so full upstream LivePortrait validation and normal pytest execution were not completed here.

## Verification

- `python3 -m compileall scripts src/avagen tests`
- Ran a direct smoke test of `run_liveportrait_inference()` against a temporary fake external `inference.py`; it completed and wrote the expected output marker file.

## Next Recommended Step

- Clone the official LivePortrait checkout into `external/LivePortrait`.
- Download official pretrained weights into `external/LivePortrait/pretrained_weights`.
- Install the upstream LivePortrait prerequisites on the machine that will run the smoke test.
- Run one real end-to-end inference command with a source image and driving video, then record the exact setup in experiment notes.

## Handoff Notes

- The core wrapper code is committed separately as `feat: implement LivePortrait wrapper`.
- Do not assume the whole LivePortrait milestone is complete yet; only the wrapper path is implemented so far.
- If implementation continues, update this file immediately so later agents do not mistake the preprocessing stubs for working pipeline code.
