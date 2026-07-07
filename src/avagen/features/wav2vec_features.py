"""wav2vec2 audio-feature extraction for audio-to-motion training.

Provides learned phonetic representations (vs hand-crafted mel/prosody), which
carry the actual speech content needed to drive mouth motion. Uses torchaudio's
WAV2VEC2_BASE bundle at 16 kHz; a middle transformer layer is used because those
layers best encode articulatory/phonetic information for lip motion.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

# torch / torchaudio are only imported lazily inside the extractor so the rest of
# the package (and tests) do not require them.

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        import torchaudio

        _MODEL = torchaudio.pipelines.WAV2VEC2_BASE.get_model().eval()
    return _MODEL


def _load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def extract_wav2vec_matrix(audio_path: str | Path, layer: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return (frames x 768) wav2vec2 features and their (frames,) time axis in seconds."""
    import torch

    samples, sample_rate = _load_wav_mono(audio_path)
    if sample_rate != 16000:
        raise ValueError(f"wav2vec2 expects 16 kHz audio, got {sample_rate} Hz for {audio_path}")
    waveform = torch.from_numpy(samples)[None, :]
    model = _get_model()
    with torch.inference_mode():
        layer_outputs, _ = model.extract_features(waveform)
    if not 0 <= layer < len(layer_outputs):
        raise ValueError(f"layer {layer} out of range for {len(layer_outputs)} wav2vec layers")
    features = layer_outputs[layer][0].cpu().numpy().astype(np.float32)  # (T, 768)
    duration = samples.shape[0] / float(sample_rate)
    n = features.shape[0]
    time_axis = (np.arange(n, dtype=np.float32) * (duration / max(n, 1))).astype(np.float32)
    return features, time_axis


def resample_to_time_grid(
    features: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    """Linearly interpolate each feature column onto target_times."""
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] < 2:
        return np.repeat(features, target_times.shape[0], axis=0).astype(np.float32)
    columns = [
        np.interp(target_times, source_times, features[:, c]).astype(np.float32)
        for c in range(features.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)
