"""Run wav-to-wav genre transfer with a trained conditional Glow checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glow_model import ConditionalGlow
from reconstruct_wav import mel_to_audio

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


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
    return ((mel - mel_min) / (mel_max - mel_min + 1e-8)).astype(np.float32)


def load_input_mel(input_path: Path, n_mels: int) -> np.ndarray:
    if input_path.suffix.lower() == ".npy":
        mel = np.load(input_path).astype(np.float32)
    elif input_path.suffix.lower() in AUDIO_EXTS:
        mel = audio_to_normalized_mel(input_path, n_mels)
    else:
        raise ValueError(f"Unsupported input type: {input_path.suffix}. Use audio or .npy.")

    if mel.ndim != 2 or mel.shape[0] != n_mels:
        raise ValueError(f"Expected mel shape [{n_mels}, T], got {mel.shape}.")
    return np.clip(mel, 0.0, 1.0).astype(np.float32)


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


def run_glow_on_full_mel(
    mel: np.ndarray,
    model: ConditionalGlow,
    source_label: int,
    target_label: int,
    crop_time: int,
    device: torch.device,
    hop_time: int,
    content_blend: float,
) -> np.ndarray:
    """Apply Glow translation to a full mel using overlap-add chunk blending."""
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

    source = torch.tensor([source_label], dtype=torch.long, device=device)
    target = torch.tensor([target_label], dtype=torch.long, device=device)
    blend = float(np.clip(content_blend, 0.0, 1.0))

    model.eval()
    for start in starts:
        chunk = mel[:, start : start + crop_time]
        original_width = chunk.shape[1]
        if original_width < crop_time:
            chunk = np.pad(chunk, ((0, 0), (0, crop_time - original_width)), mode="constant")

        x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)
        with torch.no_grad():
            z, _ = model.encode(x, source)
            translated = model.decode(z, target)
            if blend > 0.0:
                translated = (1.0 - blend) * translated + blend * x
            out = translated[0, 0].cpu().numpy().astype(np.float32)

        out = out[:, :original_width]
        weight = window_2d[:, :original_width]
        out_sum[:, start : start + original_width] += out * weight
        weight_sum[:, start : start + original_width] += weight

    if np.all(weight_sum == 0.0):
        raise ValueError("Input mel had no time frames to process.")
    return np.clip(out_sum / np.maximum(weight_sum, 1e-6), 0.0, 1.0).astype(np.float32)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Glow-based music style transfer for one audio or mel file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "glow_blues_jazz.pt")
    parser.add_argument("--direction", choices=("a2b", "b2a"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save_mel", action="store_true")
    parser.add_argument("--content_blend", type=float, default=0.25)
    parser.add_argument("--hop_time", type=int, default=None)
    parser.add_argument("--assumed_max", type=float, default=1.0)
    parser.add_argument("--n_iter", "--griffin_lim_iters", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    genre_a = ckpt.get("genre_a", "genre_a")
    genre_b = ckpt.get("genre_b", "genre_b")
    source_label, target_label = (0, 1) if args.direction == "a2b" else (1, 0)
    direction_label = f"{genre_a}_to_{genre_b}" if args.direction == "a2b" else f"{genre_b}_to_{genre_a}"

    device = choose_device(args.device)
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    n_mels = int(ckpt["n_mels"])
    crop_time = int(ckpt["crop_time"])
    hop_time = args.hop_time if args.hop_time is not None else max(1, crop_time // 2)

    input_mel = load_input_mel(args.input, n_mels)
    transferred_mel = run_glow_on_full_mel(
        input_mel,
        model,
        source_label=source_label,
        target_label=target_label,
        crop_time=crop_time,
        device=device,
        hop_time=hop_time,
        content_blend=args.content_blend,
    )

    output_wav, output_mel = output_paths(args.input, args.output, direction_label, args.save_mel)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if output_mel is not None:
        output_mel.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mel, transferred_mel)

    audio = mel_to_audio(transferred_mel, assumed_max=args.assumed_max, n_iter=args.n_iter, mel_scale="power")
    sf.write(output_wav, audio, SR)

    print(f"Checkpoint pair: {genre_a} <-> {genre_b}")
    print(f"Direction: {direction_label}")
    print(f"Content blend: {args.content_blend}")
    if output_mel is not None:
        print(f"Wrote mel: {output_mel}")
    print(f"Wrote wav: {output_wav}")


if __name__ == "__main__":
    main()
