"""Approximate inversion from a saved mel .npy back to .wav using Griffin-Lim.

This mirrors the reconstruction approach used in test/reconstruct_wav.py:
because preprocess_mel.py normalizes each mel to [0,1] without persisting the
original scale, we can only reconstruct approximately by rescaling with an
assumed maximum mel power value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from reconstruction import SR, mel_to_audio


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert one mel .npy back into an approximate .wav.")
    ap.add_argument("--input", type=Path, required=True, help="Path to a mel spectrogram .npy file.")
    ap.add_argument("--output", type=Path, required=True, help="Path to the output .wav file.")
    ap.add_argument(
        "--assumed_max",
        type=float,
        default=100.0,
        help="Rescale factor used to approximate the lost pre-normalization mel scale.",
    )
    ap.add_argument(
        "--n_iter",
        "--griffin_lim_iters",
        type=int,
        default=64,
        help="Number of Griffin-Lim iterations. Higher is slower but often cleaner.",
    )
    args = ap.parse_args()

    mel = np.load(args.input).astype(np.float32)
    try:
        audio = mel_to_audio(mel, assumed_max=args.assumed_max, n_iter=args.n_iter)
    except ValueError as exc:
        raise SystemExit(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, SR)
    print(f"Wrote {args.output} ({len(audio)} samples at {SR} Hz)")


if __name__ == "__main__":
    main()
