"""Shared mel-spectrogram to waveform reconstruction helpers."""

from __future__ import annotations

import librosa
import numpy as np

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
POWER = 2.0


def mel_to_audio(mel: np.ndarray, assumed_max: float = 100.0, n_iter: int = 64) -> np.ndarray:
    """Reconstruct an approximate waveform from a normalized mel spectrogram."""
    if mel.ndim != 2:
        raise ValueError(f"Expected a 2D mel array, got shape {mel.shape}")
    if mel.shape[0] != N_MELS:
        raise ValueError(f"Expected first dimension to be {N_MELS}, got {mel.shape[0]}")

    mel_norm = np.clip(mel.astype(np.float32), 0.0, 1.0)
    mel_power = mel_norm * assumed_max

    return librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        power=POWER,
        n_iter=n_iter,
    )
