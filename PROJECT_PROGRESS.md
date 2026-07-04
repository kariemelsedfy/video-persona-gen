# Project Progress

This is the current working-state tracker. Update it whenever repository state or next actions change.

## Status

- Date: July 3, 2026
- Phase: LivePortrait inference environment setup
- Branch: `docs/hpc-access-context`
- Overall state: `main` now contains the real LivePortrait wrapper path, and the immediate blocker is configuring Bowdoin HPC access so the first real upstream run can happen on the remote GPU environment

## Completed

- Added the top-level repository skeleton for configs, scripts, Slurm, source package modules, experiments, and docs.
- Added documentation scaffolding for ethics, benchmarks, dataset/model cards, and experiments.
- Added placeholder CLI entrypoints and package modules rather than implementing the real pipeline yet.
- Added shared session coordination files for multi-chat and multi-agent handoff.
- Replaced the LivePortrait wrapper stub with real command-building and subprocess execution against an external official checkout.
- Added a smoke-style wrapper test and updated docs/config to expect `external/LivePortrait`.
- Reviewed the Bowdoin HPC access, web portal, hardware, and GPU docs and documented the preferred remote workflow for future sessions.
- Added a shared instruction to play a local completion sound at the end of finished tasks.
- Added a local ignored `.env.hpc.local` file and attempted the first real SSH login to Bowdoin HPC using `expect`.

## Current Reality

- `scripts/run_liveportrait_inference.py` and `src/avagen/renderers/liveportrait_wrapper.py` now implement a real external-checkout wrapper path.
- `scripts/inspect_video.py`, `scripts/preprocess_dataset.py`, and `scripts/create_manifest.py` are still stubs.
- The preprocessing, face tracking, audio extraction, and model-training modules are still placeholders or explicit stubs.
- This shell currently does not have `ffmpeg`, `huggingface-cli`, `git-lfs`, or `pytest`, so full upstream LivePortrait validation and normal pytest execution were not completed here.
- There is an untracked local `LivePortrait/` directory in the working tree; treat it as local-only unless it is intentionally moved under the repo's expected `external/LivePortrait` path.
- The preferred heavy-run target is Bowdoin HPC via `moosehead.bowdoin.edu`; off-campus access requires VPN and long jobs should use Slurm on `-p gpu` with explicit `--gres`, preferably `gpu:pro6000:1` when available.
- A local ignored Bowdoin password file now exists, and this machine has `expect`, so password-driven SSH automation is technically possible here.
- The first SSH attempt reached `moosehead.bowdoin.edu` and got a password prompt, but authentication was rejected with `Permission denied (publickey,password,hostbased)`.

## Verification

- `python3 -m compileall scripts src/avagen tests`
- Ran a direct smoke test of `run_liveportrait_inference()` against a temporary fake external `inference.py`; it completed and wrote the expected output marker file.
- Fetched and inspected the Bowdoin HPC knowledge-base articles covering SSH access, the Open OnDemand portal, and GPU resource requests.
- Verified that `.env.hpc.local` contains non-empty username and password fields without obvious quoting or whitespace issues.
- Attempted SSH login to `moosehead.bowdoin.edu` using `expect`; the host was reachable but password authentication failed.

## Next Recommended Step

- Recheck the Bowdoin HPC password manually, or confirm that the Bowdoin HPC account is actually provisioned for SSH.
- If password auth should work, update `.env.hpc.local` and retry the SSH login.
- If the account is intended to use SSH keys, create or register a key and switch the workflow away from password automation.
- On the Bowdoin HPC workspace, clone or move the official LivePortrait checkout into `external/LivePortrait`, install prerequisites, and download pretrained weights.
- Run one real end-to-end inference command with a source image and driving video, then record the exact setup in experiment notes.

## Handoff Notes

- `origin/main` already includes the LivePortrait wrapper PR; this branch is only for HPC workflow and shared-context updates.
- `PROJECT_CONTEXT.md` now records the canonical Bowdoin remote-access workflow for future sessions.
- Do not assume remote authentication is configured yet; the current stored password failed against `moosehead.bowdoin.edu` even though the host was reachable.
- Future sessions should play a local completion sound when a task is finished.
