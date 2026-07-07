#!/usr/bin/env python3
"""Add wav2vec2 features to each processed clip's audio_features.npz.

wav2vec2 runs at ~50 fps; we resample onto the existing audio-feature time grid
and store it under the key ``wav2vec`` so it aligns to motion frames through the
normal aligned-sequence loader (audio_feature_names: [wav2vec]).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from avagen.data.dataset import load_processed_clip_records  # noqa: E402
from avagen.features.wav2vec_features import (  # noqa: E402
    extract_wav2vec_matrix,
    resample_to_time_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract wav2vec2 features into audio_features.npz.")
    parser.add_argument("--manifest-path", type=Path, required=True, help="Processed identity manifest.")
    parser.add_argument("--layer", type=int, default=8, help="wav2vec2 transformer layer to use (0-11).")
    parser.add_argument("--clip-id", action="append", default=[], help="Restrict to one or more clip IDs.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if 'wav2vec' already present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_processed_clip_records(args.manifest_path)
    selected = set(args.clip_id)
    for record in records:
        if selected and record.clip_id not in selected:
            continue
        if record.audio_features_path is None or record.audio_path is None:
            print(f"[wav2vec] clip_id={record.clip_id} status=skipped_no_audio")
            continue
        features_path = Path(record.audio_features_path)
        bundle = dict(np.load(features_path))
        if "wav2vec" in bundle and not args.overwrite:
            print(f"[wav2vec] clip_id={record.clip_id} status=skipped_existing dim={bundle['wav2vec'].shape}")
            continue
        target_times = np.asarray(bundle["time_axis_sec"], dtype=np.float32)
        feats, src_times = extract_wav2vec_matrix(record.audio_path, layer=args.layer)
        aligned = resample_to_time_grid(feats, src_times, target_times)
        bundle["wav2vec"] = aligned.astype(np.float32)
        bundle["wav2vec_layer"] = np.asarray(args.layer, dtype=np.int32)
        np.savez(features_path, **bundle)
        print(
            f"[wav2vec] clip_id={record.clip_id} status=completed "
            f"dim={aligned.shape} layer={args.layer} path={features_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
