# Session Summary

This file is the short end-of-session handoff. Update it every session.

## Latest Session

- Date: July 3, 2026
- Agent: Codex
- Branch: `feature/liveportrait-inference-smoke`
- Summary: replaced the LivePortrait wrapper stub with a real external-checkout runner, added smoke-test coverage, and updated docs/config for `external/LivePortrait`.
- Context update: the renderer path is now partially real while the preprocessing pipeline remains stubbed.
- Verification: `python3 -m compileall scripts src/avagen tests` and a direct smoke run of `run_liveportrait_inference()` against a temporary fake external `inference.py`.
- Open issues: this shell still lacks `ffmpeg`, `huggingface-cli`, `git-lfs`, and `pytest`, so no real upstream LivePortrait run was completed here yet.
- Next step: clone the official LivePortrait checkout, download pretrained weights, install upstream prerequisites, and run one real source-plus-driving inference.
