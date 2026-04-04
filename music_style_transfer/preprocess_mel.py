"""
Rudimentary preprocessing: audio -> normalized Mel-spectrogram -> .npy + one plot.

Put GTZAN (or any) audio under data/gtzan_subset/ — e.g. genre subfolders with .wav files.
macOS metadata files (._*, .DS_Store) are skipped automatically.

If that folder is empty, we fall back to librosa's bundled example tracks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "gtzan_subset"
OUT_DIR = ROOT / "outputs" / "mel_spectrograms"

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac"}


def is_real_audio_file(path: Path) -> bool:
    """Skip macOS AppleDouble (._file.wav), junk, and non-audio."""
    if not path.is_file():
        return False
    if path.name.startswith("._"):
        return False
    if path.name in {".DS_Store", "Thumbs.db"}:
        return False
    return path.suffix.lower() in AUDIO_EXTS


def npy_name_for(path: Path, data_root: Path) -> str:
    """e.g. blues/blues.00000.wav -> blues__blues.00000 (unique across genres)."""
    rel = path.relative_to(data_root)
    return "__".join(rel.with_suffix("").parts)


def collect_audio_paths(max_files: int | None) -> list[Path]:
    """All matching audio under DATA_DIR, sorted; optional cap for quick tests."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    found: list[Path] = []
    for p in sorted(DATA_DIR.rglob("*")):
        if not is_real_audio_file(p):
            continue
        found.append(p)
        if max_files is not None and len(found) >= max_files:
            break

    if found:
        print(f"Using {len(found)} file(s) under {DATA_DIR}")
        return found

    print(f"No audio in {DATA_DIR}; using librosa example tracks (brahms, nutcracker, fishin).")
    keys = ("brahms", "nutcracker", "fishin")
    if max_files is not None:
        keys = keys[: max_files]
    paths = [Path(librosa.util.example(k)) for k in keys]
    for p in paths:
        print(f"  - {p}")
    return paths


def audio_to_mel(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )


def normalize_mel(mel: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m, mx = mel.min(), mel.max()
    return (mel - m) / (mx - m + eps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mel-spectrogram preprocessing for GTZAN-style audio.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N files (after sort). Default: all files in data/gtzan_subset.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_root = DATA_DIR
    paths = collect_audio_paths(max_files=args.limit)

    first_stem: str | None = None
    first_mel_raw: np.ndarray | None = None
    first_sr: int | None = None

    ok = 0
    failed: list[tuple[Path, str]] = []

    for path in paths:
        try:
            if path.stat().st_size < 512:
                raise OSError("file too small (likely corrupt or empty)")
            y, sr = librosa.load(path, sr=22050, mono=True)
        except Exception as e:
            failed.append((path, repr(e)))
            print(f"SKIP {path.relative_to(data_root)}: {e}", file=sys.stderr)
            continue

        mel_raw = audio_to_mel(y, sr)
        mel_norm = normalize_mel(mel_raw).astype(np.float32)

        try:
            base = npy_name_for(path, data_root)
        except ValueError:
            base = Path(path).stem

        out_np = OUT_DIR / f"{base}_mel_norm.npy"
        np.save(out_np, mel_norm)
        ok += 1

        if ok % 50 == 0 or ok == 1:
            print(f"[{ok} saved] {out_np.name}  shape={mel_norm.shape}")

        if first_mel_raw is None:
            first_stem = base
            first_mel_raw = mel_raw
            first_sr = sr

    if first_mel_raw is None or first_sr is None or first_stem is None:
        raise SystemExit("No audio processed successfully.")

    print(f"Done. Saved {ok} .npy file(s) under {OUT_DIR}")
    if failed:
        print(f"Skipped {len(failed)} file(s) (corrupt format or unloadable).", file=sys.stderr)

    mel_db = librosa.power_to_db(first_mel_raw, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=first_sr,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        hop_length=HOP_LENGTH,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Mel spectrogram (dB): {first_stem}")
    fig.tight_layout()
    plot_path = OUT_DIR / f"{first_stem}_mel_preview.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote visualization: {plot_path}")


if __name__ == "__main__":
    main()
