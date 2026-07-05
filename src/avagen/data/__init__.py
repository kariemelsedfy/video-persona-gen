"""Data loading, preprocessing, manifests, and dataset utilities."""

from .dataset import (
    AudioMotionSequence,
    AudioMotionSequenceDataset,
    ProcessedClipDataset,
    ProcessedClipRecord,
    collate_padded_sequences,
    load_audio_features,
    load_aligned_audio_motion_sequence,
    load_clip_metadata,
    load_frame_metadata,
    load_motion_features,
    load_motion_template,
    load_processed_clip_records,
)
from .splits import SplitAssignment, apply_split_assignments, assign_clip_splits

__all__ = [
    "ProcessedClipDataset",
    "ProcessedClipRecord",
    "AudioMotionSequence",
    "AudioMotionSequenceDataset",
    "SplitAssignment",
    "apply_split_assignments",
    "assign_clip_splits",
    "collate_padded_sequences",
    "load_audio_features",
    "load_aligned_audio_motion_sequence",
    "load_clip_metadata",
    "load_frame_metadata",
    "load_motion_features",
    "load_motion_template",
    "load_processed_clip_records",
]
