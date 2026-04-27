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
) -> np.ndarray:
    """Apply the generator to a full-length mel using non-overlapping chunks."""
    width = mel.shape[1]
    chunks: list[np.ndarray] = []

    for start in range(0, width, crop_time):
        chunk = mel[:, start : start + crop_time]
        original_width = chunk.shape[1]
        if original_width < crop_time:
            chunk = np.pad(chunk, ((0, 0), (0, crop_time - original_width)), mode="constant")

        tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)
        with torch.no_grad():
            out = generator(tensor)[0, 0].cpu().numpy().astype(np.float32)
        chunks.append(out[:, :original_width])

    if not chunks:
        raise ValueError("Input mel had no time frames to process.")
    return np.concatenate(chunks, axis=1)


def mel_to_audio(
    mel: np.ndarray,
    griffin_lim_iters: int,
    energy_scale: float,
    power_exponent: float,
) -> np.ndarray:
    mel = np.clip(mel.astype(np.float32), 0.0, 1.0)
    mel_power = np.power(mel, power_exponent) * energy_scale
    return librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        power=2.0,
        n_iter=griffin_lim_iters,
    )


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
    ap.add_argument("--griffin_lim_iters", type=int, default=32)
    ap.add_argument("--energy_scale", type=float, default=1000.0)
    ap.add_argument("--power_exponent", type=float, default=2.0)
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

    input_mel = audio_to_normalized_mel(args.input_audio, n_mels=n_mels)
    transferred_mel = run_generator_on_full_mel(input_mel, generator, crop_time=crop_time, device=device)
    transferred_audio = mel_to_audio(
        transferred_mel,
        griffin_lim_iters=args.griffin_lim_iters,
        energy_scale=args.energy_scale,
        power_exponent=args.power_exponent,
    )

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
