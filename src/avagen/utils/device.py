from __future__ import annotations


def detect_torch_device(prefer_cuda: bool = True) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"
