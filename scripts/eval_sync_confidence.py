#!/usr/bin/env python3
"""Evaluate SyncNet sync-confidence for GT and predicted motion of a clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from avagen.data.dataset import load_aligned_audio_motion_sequence, load_processed_clip_records
from avagen.evaluation.sync_metrics import compute_sync_confidence, load_syncnet


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync-confidence for GT vs predicted motion.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SyncNet checkpoint.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--audio-feature-name", default="wav2vec")
    parser.add_argument("--predicted-features", type=Path, action="append", default=[],
                        help="Optional predicted_motion_features.npz (repeatable) to score vs GT.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, ckpt = load_syncnet(args.checkpoint, device=args.device)
    record = [r for r in load_processed_clip_records(args.manifest_path) if r.clip_id == args.clip_id][0]
    seq = load_aligned_audio_motion_sequence(
        record, audio_feature_names=(args.audio_feature_name,), motion_feature_name="motion_vector"
    )
    audio = seq.audio_features
    results = {"ground_truth": compute_sync_confidence(model, audio, seq.motion_features, ckpt, device=args.device)}
    for pf in args.predicted_features:
        mv = np.load(pf)["motion_vector"]
        parents = Path(pf).parents
        label = parents[2].name if len(parents) > 2 else Path(pf).stem  # e.g. huberman-flow
        results[label] = compute_sync_confidence(model, audio, mv, ckpt, device=args.device)

    print("clip:", args.clip_id)
    print(json.dumps({k: {"sync_confidence": round(v["sync_confidence"], 4),
                          "best_offset": v["best_offset"]} for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
