"""reconstruct_wav.py

Inverts the mel-spectrogram .npy files produced by preprocess_mel.py back
into listenable .wav audio using librosa's Griffin-Lim algorithm.

Standalone module + CLI:
    python reconstruct_wav.py input.npy output.wav
    python reconstruct_wav.py input.npy output.wav --sr 22050 --n_fft 2048 --n_iter 64

Known limitation
----------------
preprocess_mel.py normalises each power mel spectrogram to [0, 1] with per-file
min/max values but does NOT persist those values alongside the .npy file.
Because the original scale information is irretrievable from the .npy alone,
this module reconstructs from the saved values as relative power mels.

If you need the older experimental behavior that treated normalized values as
dB-scaled mels, pass --mel_scale db.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Install with: pip install librosa soundfile"
    )


# ---------------------------------------------------------------------------
# Core reconstruction helpers
# ---------------------------------------------------------------------------

def denormalize_mel(mel_norm: np.ndarray, assumed_max: float = 1.0) -> np.ndarray:
    """Undo [0, 1] min-max normalisation.

    preprocess_mel.py saves mels normalised to [0, 1] but discards the
    original min/max, so we cannot recover the exact original values.  We
    assume min = 0 and max = assumed_max.  For the default power-mel path,
    assumed_max is just a relative power scale.
    """
    return mel_norm * assumed_max


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Normalize reconstructed audio to a safe listening peak."""
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 1e-8:
        audio = audio * (peak / max_abs)
    return audio.astype(np.float32)


def mel_power_to_audio(
    mel_power: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    n_iter: int = 64,
) -> np.ndarray:
    """Convert a power mel spectrogram to a waveform via Griffin-Lim."""
    _ = n_mels
    mel_power = np.maximum(mel_power, 0.0)
    stft_magnitude = librosa.feature.inverse.mel_to_stft(
        mel_power,
        sr=sr,
        n_fft=n_fft,
        power=2.0,
        fmin=fmin,
        fmax=fmax if fmax is not None else sr / 2,
    )
    audio = librosa.griffinlim(
        stft_magnitude,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft,
    )
    return peak_normalize(audio)


def mel_db_to_audio(
    mel_db: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    n_iter: int = 64,
) -> np.ndarray:
    """Convert a dB-scaled mel spectrogram to a waveform via Griffin-Lim.

    Parameters
    ----------
    mel_db      : 2-D array (n_mels, time) in dB scale.
    sr          : Sample rate of the target audio.
    n_fft       : FFT window size used during preprocessing.
    hop_length  : Hop length used during preprocessing.
    n_mels      : Number of mel bands.
    fmin, fmax  : Frequency range passed to the mel filterbank.
    n_iter      : Griffin-Lim iteration count (more = cleaner, slower).

    Returns
    -------
    audio : 1-D float32 waveform.
    """
    # dB → power
    mel_power = librosa.db_to_power(mel_db)

    # Mel power → linear (STFT magnitude)
    stft_magnitude = librosa.feature.inverse.mel_to_stft(
        mel_power,
        sr=sr,
        n_fft=n_fft,
        power=2.0,
        fmin=fmin,
        fmax=fmax if fmax is not None else sr / 2,
    )

    # Griffin-Lim phase recovery
    audio = librosa.griffinlim(
        stft_magnitude,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft,
    )
    return peak_normalize(audio)


def mel_to_audio(
    mel_norm: np.ndarray,
    *,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    n_iter: int = 64,
    assumed_max: float = 1.0,
    mel_scale: str = "power",
) -> np.ndarray:
    """Convert a normalized mel spectrogram in [0,1] to a waveform.

    mel_scale="power" matches preprocess_mel.py, which saves min-max
    normalized power mels. mel_scale="db" keeps the older experimental path.
    """
    if mel_norm.ndim != 2:
        raise ValueError(
            f"Expected 2-D mel array (n_mels, time), got shape {mel_norm.shape}."
        )

    mel = denormalize_mel(mel_norm, assumed_max=assumed_max)
    if mel_scale == "power":
        return mel_power_to_audio(
            mel,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            n_iter=n_iter,
        )
    if mel_scale == "db":
        return mel_db_to_audio(
            mel,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            n_iter=n_iter,
        )
    raise ValueError(f"Unsupported mel_scale '{mel_scale}'. Use 'power' or 'db'.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconstruct(
    npy_path: str | Path,
    wav_path: str | Path,
    *,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    n_iter: int = 64,
    assumed_max: float = 1.0,
    mel_scale: str = "power",
) -> None:
    """Load a normalised mel .npy file and write a .wav reconstruction.

    Parameters
    ----------
    npy_path    : Path to the .npy file (2-D float array, shape [n_mels, time]).
    wav_path    : Destination .wav file path.
    sr          : Sample rate.
    n_fft       : FFT size (must match the value used in preprocess_mel.py).
    hop_length  : Hop length (must match the value used in preprocess_mel.py).
    n_mels      : Number of mel bands.
    fmin, fmax  : Mel filterbank frequency bounds.
    n_iter      : Griffin-Lim iterations.
    assumed_max : Approximate scale for denormalisation (see module docstring).
    mel_scale   : "power" for preprocess_mel.py outputs, or "db" for legacy use.
    """
    npy_path = Path(npy_path)
    wav_path = Path(wav_path)

    mel_norm = np.load(npy_path)  # shape: (n_mels, time_frames)
    audio = mel_to_audio(
        mel_norm,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        n_iter=n_iter,
        assumed_max=assumed_max,
        mel_scale=mel_scale,
    )

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), audio, sr)
    print(f"Saved: {wav_path}  ({len(audio)/sr:.2f}s @ {sr} Hz)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct a .wav file from a normalised mel .npy file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  type=Path, help="Path to input .npy mel spectrogram.")
    p.add_argument("output", type=Path, help="Path for the output .wav file.")
    p.add_argument("--sr",           type=int,   default=22050, help="Sample rate.")
    p.add_argument("--n_fft",        type=int,   default=2048,  help="FFT window size.")
    p.add_argument("--hop_length",   type=int,   default=512,   help="Hop length.")
    p.add_argument("--n_mels",       type=int,   default=128,   help="Number of mel bands.")
    p.add_argument("--fmin",         type=float, default=0.0,   help="Min frequency (Hz).")
    p.add_argument("--fmax",         type=float, default=None,  help="Max frequency (Hz); defaults to sr/2.")
    p.add_argument("--n_iter",       type=int,   default=64,    help="Griffin-Lim iterations.")
    p.add_argument("--assumed_max",  type=float, default=1.0, help="Approximate max scale for denormalisation.")
    p.add_argument(
        "--mel_scale",
        choices=("power", "db"),
        default="power",
        help="Use 'power' for preprocess_mel.py outputs, or 'db' for legacy experiments.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    reconstruct(
        npy_path=args.input,
        wav_path=args.output,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        n_iter=args.n_iter,
        assumed_max=args.assumed_max,
        mel_scale=args.mel_scale,
    )
