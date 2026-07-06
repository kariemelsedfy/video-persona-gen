"""Pure windowing + within-clip temporal split planning (no array deps).

Kept free of numpy so the windowing math can be unit-tested on its own and
reused by dataset construction. See ``WindowedAudioMotionDataset`` in
``avagen.data.dataset`` for the array-backed application of these plans.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowingConfig:
    """Fixed-length windowing plus within-clip temporal train/val/test split.

    Each clip's aligned frames are partitioned by time into contiguous
    ``train`` / ``val`` / ``test`` regions (so every clip contributes to all
    three splits), then sliced into fixed-length windows within the requested
    split's region. ``boundary_gap`` drops that many frames at the start of the
    ``val`` and ``test`` regions to reduce leakage from the adjacent region.
    """

    window_size: int
    stride: int
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    boundary_gap: int = 0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.boundary_gap < 0:
            raise ValueError(f"boundary_gap must be non-negative, got {self.boundary_gap}")
        if self.train_fraction < 0 or self.val_fraction < 0:
            raise ValueError("train_fraction and val_fraction must be non-negative")
        if self.train_fraction + self.val_fraction > 1.0 + 1e-9:
            raise ValueError(
                f"train_fraction + val_fraction must be <= 1.0, got "
                f"{self.train_fraction + self.val_fraction}"
            )


def temporal_split_region(
    n_frames: int, split: str, config: WindowingConfig
) -> tuple[int, int]:
    """Return the ``[start, end)`` frame region for ``split`` within one clip."""
    if n_frames <= 0:
        return (0, 0)
    n_train = int(n_frames * config.train_fraction)
    n_val = int(n_frames * config.val_fraction)
    if split == "train":
        start, end = 0, n_train
    elif split == "val":
        start, end = n_train, n_train + n_val
    elif split == "test":
        start, end = n_train + n_val, n_frames
    else:
        raise ValueError(f"Unknown split '{split}'; expected train, val, or test.")
    if config.boundary_gap > 0 and split in ("val", "test"):
        start = min(start + config.boundary_gap, end)
    return (start, end)


def plan_temporal_windows(
    n_frames: int, split: str, config: WindowingConfig
) -> list[tuple[int, int]]:
    """Enumerate ``(start, end)`` windows for ``split`` in a clip of ``n_frames``.

    Full ``window_size`` windows are stepped by ``stride``; a final end-aligned
    window is appended when the stride leaves a tail uncovered. A region shorter
    than ``window_size`` yields a single short window covering the whole region.
    """
    start, end = temporal_split_region(n_frames, split, config)
    span = end - start
    if span <= 0:
        return []
    if span < config.window_size:
        return [(start, end)]

    windows: list[tuple[int, int]] = []
    last_full_start = end - config.window_size
    cursor = start
    while cursor <= last_full_start:
        windows.append((cursor, cursor + config.window_size))
        cursor += config.stride
    if windows[-1][1] < end:
        windows.append((end - config.window_size, end))
    return windows
