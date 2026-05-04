"""Shared utilities for Glow training/transfer scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GLOW_DIR = ROOT / "glow"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GLOW_DIR) not in sys.path:
    sys.path.insert(0, str(GLOW_DIR))

from glow_model import ConditionalGlow
from reconstruct_wav import mel_to_audio, peak_normalize

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def audio_to_normalized_mel(audio_path: Path, n_mels: int) -> tuple[np.ndarray, float, float]:
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        power=2.0,
    ).astype(np.float32)
    mel_min = float(mel.min())
    mel_max = float(mel.max())
    mel_norm = ((mel - mel_min) / (mel_max - mel_min + 1e-8)).astype(np.float32)
    return mel_norm, mel_min, mel_max


def load_input_mel(input_path: Path, n_mels: int, assumed_max: float) -> tuple[np.ndarray, float]:
    if input_path.suffix.lower() == ".npy":
        mel = np.load(input_path).astype(np.float32)
        mel_max = assumed_max
    elif input_path.suffix.lower() in AUDIO_EXTS:
        mel, _mel_min, mel_max = audio_to_normalized_mel(input_path, n_mels)
    else:
        raise ValueError(f"Unsupported input type: {input_path.suffix}. Use audio or .npy.")

    if mel.ndim != 2 or mel.shape[0] != n_mels:
        raise ValueError(f"Expected mel shape [{n_mels}, T], got {mel.shape}.")
    return np.clip(mel, 0.0, 1.0).astype(np.float32), float(mel_max)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model_from_checkpoint(ckpt: dict) -> ConditionalGlow:
    return ConditionalGlow(
        n_mels=int(ckpt["n_mels"]),
        time_len=int(ckpt["crop_time"]),
        n_flows=int(ckpt["n_flows"]),
        hidden_channels=int(ckpt["hidden_channels"]),
        cond_channels=int(ckpt["cond_channels"]),
        num_classes=2,
    )


def smooth_mel(mel: np.ndarray, time_width: int, freq_width: int) -> np.ndarray:
    out = mel.astype(np.float32)
    if freq_width > 1:
        kernel = np.ones(freq_width, dtype=np.float32) / freq_width
        pad = freq_width // 2
        padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 0, padded).astype(np.float32)
    if time_width > 1:
        kernel = np.ones(time_width, dtype=np.float32) / time_width
        pad = time_width // 2
        padded = np.pad(out, ((0, 0), (pad, pad)), mode="edge")
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def print_mel_stats(name: str, mel: np.ndarray) -> None:
    high_band = mel[int(mel.shape[0] * 0.75) :, :]
    frame_delta = np.abs(np.diff(mel, axis=1)).mean() if mel.shape[1] > 1 else 0.0
    freq_delta = np.abs(np.diff(mel, axis=0)).mean() if mel.shape[0] > 1 else 0.0
    print(
        f"{name} mel stats: min={float(mel.min()):.4f} max={float(mel.max()):.4f} "
        f"mean={float(mel.mean()):.4f} high_mean={float(high_band.mean()):.4f} "
        f"time_delta={float(frame_delta):.4f} freq_delta={float(freq_delta):.4f}"
    )


def limit_high_frequency_growth(
    transferred: np.ndarray,
    original: np.ndarray,
    start_ratio: float,
    max_gain: float,
    margin: float = 0.001,
) -> np.ndarray:
    start = int(np.clip(round(transferred.shape[0] * start_ratio), 0, transferred.shape[0] - 1))
    out = transferred.copy()
    original_high = original[start:, :]
    transferred_high = out[start:, :]
    allowed = np.maximum(original_high * max_gain, original_high + margin)
    out[start:, :] = np.minimum(transferred_high, allowed)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def mel_to_audio_with_input_phase(
    mel_norm: np.ndarray,
    input_audio_path: Path,
    assumed_max: float,
    n_mels: int,
) -> np.ndarray:
    y, _ = librosa.load(input_audio_path, sr=SR, mono=True)
    mel_power = np.maximum(mel_norm * assumed_max, 0.0)
    stft_magnitude = librosa.feature.inverse.mel_to_stft(
        mel_power,
        sr=SR,
        n_fft=N_FFT,
        power=2.0,
        fmin=0.0,
        fmax=SR / 2,
    )
    input_stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    phase = np.exp(1j * np.angle(input_stft))
    width = min(stft_magnitude.shape[1], phase.shape[1])
    audio = librosa.istft(
        stft_magnitude[:, :width] * phase[:, :width],
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        length=len(y),
    )
    _ = n_mels
    return peak_normalize(audio.astype(np.float32))


def output_paths(input_path: Path, output: Path, direction_label: str, save_mel: bool) -> tuple[Path, Path | None]:
    output_is_dir = output.exists() and output.is_dir()
    if not output_is_dir and output.suffix.lower() != ".wav":
        output_is_dir = True

    stem = input_path.stem
    if stem.endswith("_mel_norm"):
        stem = stem[: -len("_mel_norm")]

    if output_is_dir:
        output_wav = output / f"{stem}_{direction_label}_glow.wav"
    else:
        output_wav = output

    output_mel = output_wav.with_suffix(".npy") if save_mel else None
    return output_wav, output_mel
