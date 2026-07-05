"""Dataset split utilities for processed clip manifests."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from avagen.data.manifests import refresh_identity_manifest


@dataclass(frozen=True)
class SplitAssignment:
    clip_id: str
    split: str


def assign_clip_splits(
    clip_ids: list[str],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> list[SplitAssignment]:
    if not clip_ids:
        return []
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Expected fractions with 0 < train_fraction and train_fraction + val_fraction < 1.")

    shuffled = sorted(set(clip_ids))
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    if total == 1:
        return [SplitAssignment(clip_id=shuffled[0], split="train")]
    if total == 2:
        return [
            SplitAssignment(clip_id=shuffled[0], split="train"),
            SplitAssignment(clip_id=shuffled[1], split="test"),
        ]

    train_cutoff = max(1, int(round(total * train_fraction)))
    val_cutoff = train_cutoff + int(round(total * val_fraction))
    if total >= 3:
        train_cutoff = min(train_cutoff, total - 2)
        val_cutoff = min(max(val_cutoff, train_cutoff + 1), total - 1)
    else:
        val_cutoff = train_cutoff

    assignments: list[SplitAssignment] = []
    for index, clip_id in enumerate(shuffled):
        if index < train_cutoff:
            split = "train"
        elif index < val_cutoff:
            split = "val"
        else:
            split = "test"
        assignments.append(SplitAssignment(clip_id=clip_id, split=split))
    return assignments


def apply_split_assignments(
    identity_dir: str | Path,
    assignments: list[SplitAssignment],
) -> tuple[Path, Path]:
    root = Path(identity_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Identity directory not found: {root}")

    split_by_clip = {assignment.clip_id: assignment.split for assignment in assignments}
    for metadata_path in sorted(root.glob("*/metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        clip_id = str(payload["clip_id"])
        if clip_id not in split_by_clip:
            continue
        payload["split"] = split_by_clip[clip_id]
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _, manifest_path, report_path = refresh_identity_manifest(root)
    return manifest_path, report_path
