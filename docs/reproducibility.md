# Reproducibility

## Environment

- **Python** ≥ 3.10 · **PyTorch** ≥ 2.4 (training/inference) · `ffmpeg` + `ffprobe`
- Trained/validated on the Bowdoin HPC LivePortrait conda env (**PyTorch + CUDA**, matplotlib 3.9)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # library + CLIs (numpy, opencv, pyyaml, tqdm)
pip install -e .[train]     # + torch, for training/inference
pip install -e .[dev]       # + pytest
```

LivePortrait is an **external checkout** (not vendored):

```bash
git clone https://github.com/KwaiVGI/LivePortrait external/LivePortrait
hf download KwaiVGI/LivePortrait --local-dir external/LivePortrait/pretrained_weights
```

## Smoke tests (no dataset required)

```bash
pytest -q                                   # 21 unit/smoke tests (transforms, shapes, flow loss)
python3 scripts/make_portfolio_figures.py   # regenerate all figures from committed logs
```
`tests/test_motion_flow.py` includes a synthetic single-window overfit check that
verifies the flow-matching loss decreases — a fast end-to-end sanity of the model.

## Full pipeline (with authorized footage)

```bash
# 1. video → face crops + audio + manifest
python scripts/preprocess_dataset.py --input clip.mp4 --identity-id my_id

# 2. GPU: extract 205-d ground-truth motion (drives frozen LivePortrait)
python scripts/extract_motion.py --config configs/extract_motion.yaml \
  --manifest-path data/processed/my_id/manifest.jsonl \
  --liveportrait-root external/LivePortrait --python-executable python

# 3. audio features + splits
python scripts/extract_wav2vec_features.py --manifest-path data/processed/my_id/manifest.jsonl
python scripts/create_splits.py --identity-id my_id --processed-root data/processed

# 4. train flow + CFG
python scripts/train_flow.py --config configs/train_flow_w2v_cfg.yaml \
  --manifest-path data/processed/my_id/manifest.jsonl --experiment-dir experiments/my_run

# 5. audio → video  (predict on CPU so the GPU is free for the renderer)
python scripts/render_predicted_motion.py --config configs/render_predicted_motion.yaml \
  --checkpoint experiments/my_run/checkpoints/best.pt \
  --model-config configs/predict_flow_w2_calm.yaml \
  --manifest-path data/processed/my_id/manifest.jsonl \
  --liveportrait-root external/LivePortrait --device cpu

# 6. objective lip-sync metric
python scripts/eval_sync_confidence.py --checkpoint <syncnet>/best.pt \
  --manifest-path data/processed/my_id/manifest.jsonl --clip-id <clip>
```

On Slurm, use the round-trip wrappers (`scripts/run_bowdoin_*_roundtrip.sh`).

## Expected directory structure

```text
data/processed/<identity>/<clip_id>/{audio.wav, face_crops/, motion_template.pkl,
                                     motion_features.npz, audio_features.npz, metadata.json}
experiments/<name>/{checkpoints/{best,last}.pt, metrics/{summary.json, train_history.jsonl},
                    config.resolved.json}
```

## Checkpoints & data

Not committed (see `.gitignore`): checkpoints (15–176 MB), raw/processed data,
`external/`, media. Regenerate by re-running the pipeline; the committed
`docs/assets/experiment_logs/` are sufficient to reproduce every figure without HPC.

## Determinism, hardware, runtime, common failures

- **Seeds:** `utils/seed.py`; configs set `seed`.
- **Hardware:** CPU works for preprocessing/prediction; a GPU (≥ 11 GB) is needed
  for motion extraction and rendering.
- **Runtime:** extraction ~6.4 h / 16 clips; flow+CFG training ~22 min; render ~5 min/clip.
- **Common failures:** LivePortrait `cudaErrorDevicesUnavailable` on
  Exclusive_Process GPUs → predict on **CPU** (`--device cpu`); wav2vec2 download
  quota error → set `TORCH_HOME`/`HF_HOME` to scratch; audio drift → alignment uses
  true `record.fps`. See [compute.md](compute.md).
