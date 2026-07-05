# Project Progress

This is the current working-state tracker. Update it whenever repository state or next actions change.

## Status

- Date: July 4, 2026
- Phase: Bowdoin storage hardening and preprocessing pipeline implementation
- Branch: `docs/hpc-access-context`
- Overall state: the Bowdoin remote environment can complete a real upstream LivePortrait inference job while reusing persistent weights from `/mnt/hpc/tmp/kelsedfy/video-persona-gen/liveportrait_weights`, the durable scratch layout is now verified end to end, and the repo contains a first real preprocessing path that has already passed a Bowdoin smoke test on a synthetic audio-backed sample clip

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
- Submitted and debugged multiple Bowdoin Slurm jobs for LivePortrait environment setup and Hugging Face weight download.
- Reconnected through the Bowdoin VPN, recovered the previously missing Hugging Face and Slurm stderr, and confirmed the upstream `readme.md` download command is wrong for the current CLI because it only downloads `README.md` and `docs`.
- Confirmed Bowdoin home storage is already at the hard limit (`20 GiB` soft quota, `25 GiB` hard limit / `25600M` used), which causes direct `hf download` runs into the home workspace to fail and can also crash inference when Rich flushes to a home-backed Slurm log.
- Completed the first real upstream LivePortrait GPU run on Bowdoin by staging the checkout, weights, logs, and outputs entirely under node-local `/tmp` inside a Slurm job.
- Added a tracked Bowdoin helper job at `slurm/liveportrait_infer_tmp.sbatch` that captures the working node-local `/tmp` inference recipe and supports optional output persistence through `PERSIST_OUTPUT_DIR`.
- Added repo-local Bowdoin automation scripts that can submit a remote LivePortrait job, wait for a success status, download the output MP4s and logs back into the local repo, and then release the remote hold.
- Verified that `/mnt/hpc/tmp/kelsedfy` is writable, currently almost empty, and backed by a filesystem with about `32T` free, then updated the Bowdoin job scripts to use `/mnt/hpc/tmp/kelsedfy/video-persona-gen` as the durable storage root.
- Replaced the preprocessing stubs with a minimal real pipeline built on `ffprobe`, `ffmpeg`, and OpenCV: clip inspection, audio extraction, frame sampling, Haar-cascade face crops with center-crop fallback, metadata emission, and identity-level manifest/report generation.
- Added a pure-Python manifest smoke test at `tests/test_manifest_generation.py`.
- Synced the preprocessing code to the Bowdoin repo and ran a real Slurm smoke test on the `main` partition against a synthetic audio-backed version of `external/LivePortrait/assets/examples/driving/d0.mp4`.
- Verified the updated persistent-storage LivePortrait round-trip flow with two Bowdoin jobs: the first populated `/mnt/hpc/tmp/kelsedfy/video-persona-gen/liveportrait_weights`, and the second explicitly reused those persisted weights instead of redownloading them.

## Current Reality

- `scripts/run_liveportrait_inference.py` and `src/avagen/renderers/liveportrait_wrapper.py` now implement a real external-checkout wrapper path.
- `scripts/inspect_video.py`, `scripts/preprocess_dataset.py`, and `scripts/create_manifest.py` now call real package code instead of returning skeleton JSON.
- The preprocessing path now writes `audio.wav`, `face_crops/`, `frame_metadata.json`, `metadata.json`, `manifest.jsonl`, and `dataset_report.json`.
- Face tracking is still a minimal OpenCV Haar-cascade baseline with fallback cropping; landmarks, head pose, expressions, and motion templates are still unimplemented.
- This shell currently does not have `ffmpeg`, `huggingface-cli`, `git-lfs`, or `pytest`, so full upstream LivePortrait validation and normal pytest execution were not completed here.
- Bowdoin now has a confirmed durable layout for this project:
  - weights: `/mnt/hpc/tmp/kelsedfy/video-persona-gen/liveportrait_weights`
  - fetched inference runs: `/mnt/hpc/tmp/kelsedfy/video-persona-gen/liveportrait_runs/<jobid>/`
  - processed data: `/mnt/hpc/tmp/kelsedfy/video-persona-gen/data/processed/`
- There is an untracked local `LivePortrait/` directory in the working tree; treat it as local-only unless it is intentionally moved under the repo's expected `external/LivePortrait` path.
- The preferred heavy-run target is Bowdoin HPC via `moosehead.bowdoin.edu`; off-campus access requires VPN and long jobs should use Slurm on `-p gpu` with explicit `--gres`, preferably `gpu:pro6000:1` when available.
- A local ignored Bowdoin password file now exists, and this machine has `expect`, so password-driven SSH automation is technically possible here.
- Password-driven SSH to `moosehead.bowdoin.edu` now works from this machine using `.env.hpc.local` and `expect`.
- A reusable local skill now exists at `~/.codex/skills/bowdoin-hpc-ssh`, with a tested script at `scripts/run_bowdoin_hpc_command.sh`.
- The remote Bowdoin code workspace now exists at `/home/kelsedfy/video-persona-gen`.
- The upstream renderer checkout now exists at `/home/kelsedfy/video-persona-gen/external/LivePortrait`.
- `moosehead` is only for shell and Slurm orchestration; invoking `python3` there prints a warning to use Slurm or an interactive machine instead.
- Bowdoin exposes `/usr/bin/ffmpeg` and modules including `miniconda3`, `python3.11`, `python3.11.8`, `cuda-12.8.1`, `cuda-12.9.1`, and `cuda-13.1`.
- A remote conda environment now exists at `/home/kelsedfy/video-persona-gen/.conda/liveportrait`.
- On Bowdoin `pro6000` nodes, `nvidia-smi` reports `NVIDIA RTX PRO 6000 Blackwell Server Edition` with driver `610.43.02`, and `nvcc -V` under `cuda-12.8.1` reports CUDA `12.8`.
- The remote environment now has:
  - `torch==2.11.0+cu128`
  - `torchvision==0.26.0+cu128`
  - `torchaudio==2.11.0+cu128`
  - `transformers==4.38.0`
  - `onnxruntime-gpu==1.18.0`
  - `huggingface_hub==0.36.2`
- The initial setup job failed only at the deprecated `huggingface-cli download` step.
- A follow-up repair job restored `huggingface_hub==0.36.2` and `pip check` passed.
- The working weights command is `hf download KlingTeam/LivePortrait --local-dir ...`; the upstream `huggingface-cli download ... "README.md" "docs"` form only fetched those paths on the current CLI.
- Direct home-side weight downloads still fail because the Bowdoin account is already at its hard storage limit.
- The home-side `external/LivePortrait/pretrained_weights` tree is only a partial artifact with docs and incomplete cache files; it is not a usable weight directory.
- A home-backed Slurm log can also fail mid-inference with `OSError: [Errno 122] Disk quota exceeded`, even after the node-local model download succeeds.
- A node-local `/tmp` Slurm flow now works around both issues and completed a full upstream inference job on `moose63` with `--gres=gpu:rtx3080:1`.
- `/mnt/hpc/tmp/kelsedfy` is writable and large enough to serve as the durable scratch root for this project, and the tracked Bowdoin scripts now default to `/mnt/hpc/tmp/<user>/video-persona-gen`.
- `onnxruntime-gpu` emitted CUDA-provider load errors on the `rtx3080` run because `libcublasLt.so.11` was missing, but the overall job still completed successfully.
- The verified round-trip scripts are:
  - `scripts/run_bowdoin_hpc_command.sh`
  - `scripts/fetch_bowdoin_liveportrait_output.sh`
  - `scripts/run_bowdoin_liveportrait_roundtrip.sh`
- The verified local sample output path is `outputs/bowdoin_liveportrait/verified-sample/output/`, containing `s0--d0.mp4` and `s0--d0_concat.mp4`.

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
- Ran a short `pro6000` GPU probe job and confirmed the target GPU/driver/CUDA combination:
  - GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
  - Driver: `610.43.02`
  - CUDA toolkit: `12.8`
- Confirmed from the official PyTorch site on July 3, 2026 that stable Linux pip installs support CUDA `12.8`, and used a `cu128` PyTorch install path as an inference from that official selector.
- Ran a main-partition setup job that successfully created the remote conda env and installed the LivePortrait Python dependencies before failing at `huggingface-cli download`.
- Ran a repair job that verified `hf` exists in the env at `/home/kelsedfy/video-persona-gen/.conda/liveportrait/bin/hf`.
- Confirmed from the upstream remote `readme.md` that the published weights command still appends `README.md` and `docs` as positional paths after `huggingface-cli download`.
- Captured `outputs/logs/liveportrait_weights_fix_63741.out`, which showed the corrected `hf download` path working until it hit `OSError: [Errno 122] Disk quota exceeded`.
- Ran `quota -s` on Bowdoin and confirmed the account is already at the hard limit (`25600M` used, `20480M` quota, `25600M` limit).
- Inspected `sinfo -p gpu -o '%N %G %t'` and used the idle `rtx3080` capacity for the first successful inference run.
- Verified that job `63744` reached real inference and failed only because Rich flushed to a home-backed Slurm log after reporting `The animated video consists of 78 frames`.
- Verified that job `63745` completed with exit code `0` after running the Hugging Face download and `python inference.py -s assets/examples/source/s0.jpg -d assets/examples/driving/d0.mp4` entirely under node-local `/tmp`.
- Verified the full submit-and-download workflow with job `63748`: the remote job completed successfully, the local fetch script pulled the output MP4s and logs into `outputs/bowdoin_liveportrait/verified-sample/`, and the final local artifacts were:
  - `outputs/bowdoin_liveportrait/verified-sample/output/s0--d0.mp4`
  - `outputs/bowdoin_liveportrait/verified-sample/output/s0--d0_concat.mp4`
- Verified on Bowdoin that `/mnt/hpc/tmp/kelsedfy` is writable, has mode `drwxrwxr-x`, and sits on `gluster1.bowdoin.edu:/gv0` with roughly `32T` free.
- Verified local shell syntax with `bash -n` for the updated Bowdoin scripts and `slurm` templates.
- Verified Python compilation with `python3 -m compileall scripts src/avagen tests`.
- Ran a local manifest smoke test against synthetic metadata using `PYTHONPATH=src python3 scripts/create_manifest.py --identity-id demo_id --processed-root <tmpdir>/data/processed`.
- Ran Bowdoin Slurm job `63750` on the `main` partition, which:
  - generated `/mnt/hpc/tmp/kelsedfy/video-persona-gen/smoke/preprocess/d0_with_tone.mp4` by adding a synthetic sine-wave audio track to the LivePortrait driving sample
  - confirmed `scripts/inspect_video.py` saw `has_audio=true`, `fps=25.0`, and `num_frames=78`
  - confirmed `scripts/preprocess_dataset.py` produced `78` face crops with `face_detection_rate=1.0`
  - wrote processed outputs under `/mnt/hpc/tmp/kelsedfy/video-persona-gen/data/processed/smoke_preprocess/`
- Ran Bowdoin jobs `63751` and `63752` through `scripts/run_bowdoin_liveportrait_roundtrip.sh`:
  - `63751` completed successfully and populated `/mnt/hpc/tmp/kelsedfy/video-persona-gen/liveportrait_weights`
  - `63752` completed successfully and `outputs/bowdoin_liveportrait/persistent-storage-smoke-2/logs/hf.log` showed `Reusing persisted weights at /mnt/hpc/tmp/kelsedfy/video-persona-gen/liveportrait_weights`
  - `63752` status recorded `persist_output_dir=/mnt/hpc/tmp/kelsedfy/video-persona-gen/liveportrait_runs/63752`
  - local fetched output sizes for `63752` were about `107209` bytes for `s0--d0.mp4` and `308910` bytes for `s0--d0_concat.mp4`

## Next Recommended Step

- Run the first real third-party or self-recorded talking-head clip through `scripts/preprocess_dataset.py`, using `/mnt/hpc/tmp/kelsedfy/video-persona-gen/data/processed` as the default Bowdoin processed-data root.
- Decide whether Bowdoin preprocessing should run directly in the existing repo env or in a dedicated preprocessing env separate from the LivePortrait env.
- Keep `/mnt/hpc/tmp/<user>/video-persona-gen/liveportrait_weights` as the canonical Bowdoin weight cache and stop redownloading weights per run.
- Decide whether to keep password-based automation or convert the verified workflow to SSH keys for a cleaner long-term setup.
- Investigate the `onnxruntime-gpu` CUDA-provider mismatch (`libcublasLt.so.11` missing) so the ONNX subcomponents use GPU acceleration on Bowdoin instead of falling back noisily.
- After one clip preprocesses cleanly, implement the next layer of derived features: landmarks, head pose, expressions, or LivePortrait motion-template extraction.

## Handoff Notes

- `origin/main` already includes the LivePortrait wrapper PR; this branch is only for HPC workflow and shared-context updates.
- `PROJECT_CONTEXT.md` now records the canonical Bowdoin remote-access workflow for future sessions.
- Remote authentication is now working from this machine with `.env.hpc.local`, and the reusable local skill lives at `~/.codex/skills/bowdoin-hpc-ssh`.
- The remote code workspace and conda environment are ready, and the corrected weights command is `hf download KlingTeam/LivePortrait --local-dir ...`.
- The real blocking issue is now Bowdoin home storage: direct downloads into the home workspace and home-backed Slurm logs can fail with `Disk quota exceeded`.
- The durable Bowdoin storage root is now `/mnt/hpc/tmp/kelsedfy/video-persona-gen`.
- The tracked short-term runtime workaround still lives at `slurm/liveportrait_infer_tmp.sbatch`, and it now has a verified persistent weight cache at `/mnt/hpc/tmp/<user>/video-persona-gen/liveportrait_weights`.
- The verified local submit/fetch entrypoint now lives at `scripts/run_bowdoin_liveportrait_roundtrip.sh`.
- The preprocessing entrypoints are no longer stubs; `scripts/preprocess_dataset.py` now emits real clip directories and refreshes the identity manifest by default.
- Bowdoin Slurm job IDs from this session:
  - `63713`: successful `pro6000` GPU probe
  - `63715`: main-partition setup job; env and dependencies installed, failed at deprecated `huggingface-cli`
  - `63716`: repair job; restored `huggingface_hub==0.36.2`, `pip check` passed
  - `63717`: confirmed `hf` CLI exists in the env
  - `63718`: prior `hf download` attempt
  - `63741`: corrected `hf download` into home workspace; failed with `Disk quota exceeded`
  - `63744`: first real inference attempt on `moose63`; reached frame generation and failed only because the home-backed Slurm log hit quota
  - `63745`: node-local `/tmp` inference run on `moose63`; completed successfully with exit code `0`
  - `63747`: first round-trip wrapper submission; remote run succeeded but exposed local fetch wrapper bugs
  - `63748`: verified round-trip workflow; remote run succeeded and local MP4/log download succeeded
- Additional verified Bowdoin jobs from this session:
  - `63750`: Slurm smoke test for the new preprocessing pipeline on a synthetic audio-backed sample clip
  - `63751`: first persistent-storage LivePortrait run; created the durable weight cache and persisted run artifacts under `/mnt/hpc/tmp/kelsedfy/video-persona-gen/liveportrait_runs/63751`
  - `63752`: second persistent-storage LivePortrait run; reused the durable weight cache and persisted run artifacts under `/mnt/hpc/tmp/kelsedfy/video-persona-gen/liveportrait_runs/63752`
- The current remote recipe is: keep the existing Bowdoin env in home, stage the LivePortrait checkout copy under node-local `/tmp` for runtime, and persist reusable weights plus fetched artifacts under `/mnt/hpc/tmp/<user>/video-persona-gen`.
- Future sessions should play a local completion sound when a task is finished.
