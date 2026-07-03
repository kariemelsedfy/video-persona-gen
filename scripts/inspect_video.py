#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect basic metadata for one or more videos.")
    parser.add_argument("videos", nargs="+", help="Path(s) to input video files.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "status": "skeleton",
        "script": "inspect_video",
        "videos": args.videos,
        "next_step": "Implement avagen.utils.video.inspect_video.",
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
