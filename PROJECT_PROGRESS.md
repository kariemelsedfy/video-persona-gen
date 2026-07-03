# Project Progress

This is the current working-state tracker. Update it whenever repository state or next actions change.

## Status

- Date: July 2, 2026
- Phase: repository scaffold only
- Branch: `docs/session-context-files`
- Overall state: the repo structure exists, but the actual preprocessing and LivePortrait pipeline logic are still stubs

## Completed

- Added the top-level repository skeleton for configs, scripts, Slurm, source package modules, experiments, and docs.
- Added documentation scaffolding for ethics, benchmarks, dataset/model cards, and experiments.
- Added placeholder CLI entrypoints and package modules rather than implementing the real pipeline yet.
- Added shared session coordination files for multi-chat and multi-agent handoff.
- Added a persistent workflow rule that significant working changes should be committed as the work progresses.

## Current Reality

- `scripts/inspect_video.py`, `scripts/preprocess_dataset.py`, `scripts/create_manifest.py`, and `scripts/run_liveportrait_inference.py` currently expose the intended CLI shape only.
- The corresponding modules under `src/avagen/` are placeholders or explicit stubs.
- No real preprocessing, face tracking, audio extraction, LivePortrait inference, or model training has been implemented yet.

## Verification

- `python3 -m compileall scripts src/avagen`
- Ran each stub CLI once to confirm the skeleton executes without import/runtime failures.

## Next Recommended Step

- Pick one concrete milestone and implement it fully.
- Recommended first implementation target: the LivePortrait inference wrapper or the one-clip preprocessing smoke test.

## Handoff Notes

- Keep future commits atomic. The scaffold should be committed separately from any real preprocessing or renderer implementation.
- Significant work should be committed during the session, not deferred into one large end-of-session commit, unless the user explicitly requests otherwise.
- If implementation starts, update this file immediately so later agents do not mistake stubs for working pipeline code.
