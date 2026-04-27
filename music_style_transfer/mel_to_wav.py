"""Approximate inversion from a saved mel .npy back to .wav using Griffin-Lim.

This works best for the normalized mel files produced by preprocess_mel.py and
the converted mel files produced by convert_mel.py. Because the preprocessing
step min-max normalizes each spectrogram independently, the original loudness
scale is not preserved, so reconstruction is only an approximation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert one mel .npy back into an approximate .wav.")
    ap.add_argument("--input", type=Path, required=True, help="Path to a mel spectrogram .npy file.")
    ap.add_argument("--output", type=Path, required=True, help="Path to the output .wav file.")
    ap.add_argument(
        "--griffin_lim_iters",
        type=int,
        default=32,
        help="Number of Griffin-Lim iterations. Higher is slower but often cleaner.",
    )
    ap.add_argument(
        "--energy_scale",
        type=float,
        default=1000.0,
        help="Scales normalized mel values before inversion. Tune if output is too quiet/noisy.",
    )
    ap.add_argument(
        "--power_exponent",
        type=float,
        default=2.0,
        help="Expands normalized mel contrast before inversion. Values >1 emphasize peaks.",
    )
    args = ap.parse_args()

    mel = np.load(args.input).astype(np.float32)
    if mel.ndim != 2:
        raise SystemExit(f"Expected a 2D mel array, got shape {mel.shape}")
    if mel.shape[0] != N_MELS:
        raise SystemExit(f"Expected first dimension to be {N_MELS}, got {mel.shape[0]}")

    # These .npy files are normalized to [0,1], so we rebuild an approximate
    # non-negative mel power spectrogram before applying Griffin-Lim inversion.
    mel = np.clip(mel, 0.0, 1.0)
    mel_power = np.power(mel, args.power_exponent) * args.energy_scale

    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        power=2.0,
        n_iter=args.griffin_lim_iters,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, SR)
    print(f"Wrote {args.output} ({len(audio)} samples at {SR} Hz)")


if __name__ == "__main__":
    main()
