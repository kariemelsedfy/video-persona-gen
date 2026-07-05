#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from avagen.data.manifests import refresh_identity_manifest
from avagen.data.preprocessing import PreprocessConfig, preprocess_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess one or more talking-head videos into clip folders.")
    parser.add_argument("--input", nargs="+", required=True, help="Input video path(s).")
    parser.add_argument("--identity-id", required=True, help="Target identity id, for example self_001.")
    parser.add_argument("--clip-id", help="Optional explicit clip id when preprocessing exactly one input.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed dataset root.",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="Target output FPS for frame extraction.")
    parser.add_argument("--audio-sample-rate", type=int, default=16000, help="Target audio sample rate.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split label.")
    parser.add_argument("--face-margin", type=float, default=0.2, help="Relative crop expansion around the face.")
    parser.add_argument(
        "--disable-center-crop-fallback",
        action="store_true",
        help="Fail if a face is not detected and no previous box is available.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing clip directory.")
    parser.add_argument(
        "--skip-manifest-refresh",
        action="store_true",
        help="Only preprocess clips and do not regenerate the identity manifest.",
    )
    return parser.parse_args()


def _slugify_clip_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "clip"


def main() -> int:
    args = parse_args()
    identity_dir = args.output_root / args.identity_id
    if args.clip_id and len(args.input) != 1:
        raise ValueError("--clip-id may only be provided when preprocessing exactly one input clip.")

    results = []
    for index, raw_input in enumerate(args.input):
        input_path = Path(raw_input)
        clip_id = args.clip_id if args.clip_id else _slugify_clip_id(input_path.stem)
        if not args.clip_id and len(args.input) > 1:
            clip_id = f"{clip_id}_{index:03d}"

        results.append(
            preprocess_clip(
                PreprocessConfig(
                    input_path=input_path,
                    identity_id=args.identity_id,
                    clip_id=clip_id,
                    output_root=args.output_root,
                    target_fps=args.fps,
                    audio_sample_rate=args.audio_sample_rate,
                    split=args.split,
                    face_margin=args.face_margin,
                    allow_center_crop_fallback=not args.disable_center_crop_fallback,
                    overwrite=args.overwrite,
                )
            )
        )

    payload = {
        "identity_dir": str(identity_dir.resolve()),
        "processed_clips": results,
    }
    if not args.skip_manifest_refresh:
        records, manifest_output, report_output = refresh_identity_manifest(identity_dir)
        payload["manifest_path"] = str(manifest_output)
        payload["report_path"] = str(report_output)
        payload["num_manifest_records"] = len(records)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
