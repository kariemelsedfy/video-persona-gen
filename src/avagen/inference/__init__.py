"""Inference pipelines."""

from .generate_motion import load_motion_predictor, predict_motion_for_manifest

__all__ = ["load_motion_predictor", "predict_motion_for_manifest"]
