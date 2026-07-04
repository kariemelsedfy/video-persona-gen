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
- Corrected the Bowdoin password, verified successful SSH access to `moosehead`, and created a reusable local Codex skill for Bowdoin HPC SSH.
- Staged the remote Bowdoin code workspace by cloning this repo to `/home/kelsedfy/video-persona-gen` and upstream LivePortrait to `/home/kelsedfy/video-persona-gen/external/LivePortrait`.

## Current Reality

- `scripts/run_liveportrait_inference.py` and `src/avagen/renderers/liveportrait_wrapper.py` now implement a real external-checkout wrapper path.
- `scripts/inspect_video.py`, `scripts/preprocess_dataset.py`, and `scripts/create_manifest.py` are still stubs.
- The preprocessing, face tracking, audio extraction, and model-training modules are still placeholders or explicit stubs.
- This shell currently does not have `ffmpeg`, `huggingface-cli`, `git-lfs`, or `pytest`, so full upstream LivePortrait validation and normal pytest execution were not completed here.
- There is an untracked local `LivePortrait/` directory in the working tree; treat it as local-only unless it is intentionally moved under the repo's expected `external/LivePortrait` path.
- The preferred heavy-run target is Bowdoin HPC via `moosehead.bowdoin.edu`; off-campus access requires VPN and long jobs should use Slurm on `-p gpu` with explicit `--gres`, preferably `gpu:pro6000:1` when available.
- A local ignored Bowdoin password file now exists, and this machine has `expect`, so password-driven SSH automation is technically possible here.
- Password-driven SSH to `moosehead.bowdoin.edu` now works from this machine using `.env.hpc.local` and `expect`.
- A reusable local skill now exists at `~/.codex/skills/bowdoin-hpc-ssh`, with a tested script at `scripts/run_bowdoin_hpc_command.sh`.
- The remote Bowdoin code workspace now exists at `/home/kelsedfy/video-persona-gen`.
- The upstream renderer checkout now exists at `/home/kelsedfy/video-persona-gen/external/LivePortrait`.
- `moosehead` is only for shell and Slurm orchestration; invoking `python3` there prints a warning to use Slurm or an interactive machine instead.
- Bowdoin exposes `/usr/bin/ffmpeg` and modules including `miniconda3`, `python3.11`, `python3.11.8`, `cuda-12.8.1`, `cuda-12.9.1`, and `cuda-13.1`.

## Verification

- `python3 -m compileall scripts src/avagen tests`
- Ran a direct smoke test of `run_liveportrait_inference()` against a temporary fake external `inference.py`; it completed and wrote the expected output marker file.
- Fetched and inspected the Bowdoin HPC knowledge-base articles covering SSH access, the Open OnDemand portal, and GPU resource requests.
- Verified that `.env.hpc.local` contains non-empty username and password fields without obvious quoting or whitespace issues.
- Attempted SSH login to `moosehead.bowdoin.edu` using `expect`; the host was reachable but password authentication failed.
- Retried SSH after correcting the password; login succeeded, `hostname` returned `moosehead`, and `sbatch` was available.
- Forward-tested `~/.codex/skills/bowdoin-hpc-ssh/scripts/run_bowdoin_hpc_command.sh` successfully against `moosehead`.
- Confirmed GitHub access from Bowdoin HPC for both `kariemelsedfy/video-persona-gen` and `KlingAIResearch/LivePortrait`.
- Cloned both repositories on Bowdoin HPC and verified the checked-out HEADs:
  - project repo: `1943167` on `main`
  - LivePortrait repo: `9b294b3` on `main`
- Confirmed `/usr/bin/ffmpeg` exists on Bowdoin HPC and that module listings include `miniconda3` and multiple CUDA versions.

## Next Recommended Step

- Decide whether to keep password-based automation or convert the verified workflow to SSH keys for a cleaner long-term setup.
- Choose the runtime path for Python work: Slurm GPU job or approved interactive machine, not `moosehead`.
- On the Bowdoin HPC workspace, load `miniconda3` or another Python module, create the LivePortrait environment, and install upstream dependencies.
- Download pretrained weights into `/home/kelsedfy/video-persona-gen/external/LivePortrait/pretrained_weights`.
- Run one real end-to-end inference command with a source image and driving video, then record the exact setup in experiment notes.

## Handoff Notes

- `origin/main` already includes the LivePortrait wrapper PR; this branch is only for HPC workflow and shared-context updates.
- `PROJECT_CONTEXT.md` now records the canonical Bowdoin remote-access workflow for future sessions.
- Remote authentication is now working from this machine with `.env.hpc.local`, and the reusable local skill lives at `~/.codex/skills/bowdoin-hpc-ssh`.
- The remote code workspace is ready, but environment creation and weight download still need to happen on a non-headnode execution path.
- Future sessions should play a local completion sound when a task is finished.
