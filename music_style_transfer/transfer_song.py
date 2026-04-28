"""End-to-end music style transfer for one song.

Pipeline:
1. Load an input audio file
2. Convert it to the normalized mel format used by this project
3. Run the trained CycleGAN generator on sliding time chunks
4. Reconstruct an approximate waveform with Griffin-Lim
5. Save both the transferred mel and the output wav

This is designed to work with the checkpoint format produced by
train_cycle_gan.py in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from cycle_gan import Generator
from reconstruct_wav import mel_to_audio

ROOT = Path(__file__).resolve().parent

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def audio_to_normalized_mel(audio_path: Path, n_mels: int) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        power=2.0,
    )
    mel = mel.astype(np.float32)
    mel_min = float(mel.min())
    mel_max = float(mel.max())
    mel_norm = (mel - mel_min) / (mel_max - mel_min + 1e-8)
    return mel_norm.astype(np.float32)


def run_generator_on_full_mel(
    mel: np.ndarray,
    generator: Generator,
    crop_time: int,
    device: torch.device,
    hop_time: int,
) -> np.ndarray:
    """Apply the generator to a full-length mel using overlap-add chunk blending."""
    width = mel.shape[1]
    if hop_time <= 0:
        raise ValueError("hop_time must be positive.")
    if hop_time > crop_time:
        hop_time = crop_time

    starts = list(range(0, max(1, width - crop_time + 1), hop_time))
    if not starts or starts[-1] != max(0, width - crop_time):
        starts.append(max(0, width - crop_time))

    out_sum = np.zeros((mel.shape[0], width), dtype=np.float32)
    weight_sum = np.zeros((1, width), dtype=np.float32)
    window = np.hanning(crop_time).astype(np.float32)
    if crop_time == 1:
        window[:] = 1.0
    else:
        window = np.maximum(window, 1e-3)
    window_2d = window[None, :]

    for start in starts:
        chunk = mel[:, start : start + crop_time]
        original_width = chunk.shape[1]
        if original_width < crop_time:
            chunk = np.pad(chunk, ((0, 0), (0, crop_time - original_width)), mode="constant")

        tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)
        with torch.no_grad():
            out = generator(tensor)[0, 0].cpu().numpy().astype(np.float32)
        out = out[:, :original_width]
        weight = window_2d[:, :original_width]
        out_sum[:, start : start + original_width] += out * weight
        weight_sum[:, start : start + original_width] += weight

    if np.all(weight_sum == 0.0):
        raise ValueError("Input mel had no time frames to process.")
    return out_sum / np.maximum(weight_sum, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end music style transfer for one audio file.")
    ap.add_argument(
        "--input",
        "--input_audio",
        dest="input_audio",
        type=Path,
        required=True,
        help="Path to the source audio file.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "cycle_gan.pt",
        help="Path to a checkpoint from train_cycle_gan.py",
    )
    ap.add_argument(
        "--direction",
        choices=("a2b", "b2a"),
        required=True,
        help="Use the generator that maps genre_a -> genre_b or genre_b -> genre_a.",
    )
    ap.add_argument(
        "--output",
        "--output_dir",
        dest="output",
        type=Path,
        required=True,
        help="Output directory or a specific output .wav path.",
    )
    ap.add_argument(
        "--output_mel",
        type=Path,
        default=None,
        help="Optional path for the transferred mel .npy. If omitted, no .npy is saved.",
    )
    ap.add_argument("--assumed_max", type=float, default=100.0)
    ap.add_argument("--n_iter", "--griffin_lim_iters", type=int, default=64)
    ap.add_argument(
        "--hop_time",
        type=int,
        default=None,
        help="Chunk hop in mel frames for overlap-add inference. Defaults to crop_time // 2.",
    )
    ap.add_argument(
        "--save_mel",
        action="store_true",
        help="Also save the transferred mel spectrogram as a .npy file inside output_dir.",
    )
    args = ap.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    n_mels = int(ckpt["n_mels"])
    crop_time = int(ckpt["crop_time"])
    genre_a = ckpt.get("genre_a", "genre_a")
    genre_b = ckpt.get("genre_b", "genre_b")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU.")
        device = torch.device("cpu")

    generator = Generator(n_mels, crop_time)
    key = "G_AB" if args.direction == "a2b" else "G_BA"
    generator.load_state_dict(ckpt[key])
    generator.to(device)
    generator.eval()

    hop_time = args.hop_time if args.hop_time is not None else max(1, crop_time // 2)
    input_mel = audio_to_normalized_mel(args.input_audio, n_mels=n_mels)
    transferred_mel = run_generator_on_full_mel(
        input_mel,
        generator,
        crop_time=crop_time,
        device=device,
        hop_time=hop_time,
    )
    transferred_audio = mel_to_audio(transferred_mel, assumed_max=args.assumed_max, n_iter=args.n_iter)

    direction_label = f"{genre_a}_to_{genre_b}" if args.direction == "a2b" else f"{genre_b}_to_{genre_a}"
    input_stem = args.input_audio.stem
    output_target = args.output
    output_is_dir = output_target.exists() and output_target.is_dir()
    if not output_is_dir and output_target.suffix.lower() != ".wav":
        output_is_dir = True

    if output_is_dir:
        output_dir = output_target
        output_wav = output_dir / f"{input_stem}_{direction_label}.wav"
    else:
        output_wav = output_target
        output_dir = output_wav.parent

    output_mel = args.output_mel
    if output_mel is None and args.save_mel:
        output_mel = output_wav.with_suffix(".npy")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if output_mel is not None:
        output_mel.parent.mkdir(parents=True, exist_ok=True)

    if output_mel is not None:
        np.save(output_mel, transferred_mel.astype(np.float32))
    sf.write(output_wav, transferred_audio, SR)

    print(f"Checkpoint pair: {genre_a} <-> {genre_b}")
    print(f"Direction: {direction_label}")
    if output_mel is not None:
        print(f"Wrote mel: {output_mel}")
    print(f"Wrote wav: {output_wav}")


if __name__ == "__main__":
    main()
