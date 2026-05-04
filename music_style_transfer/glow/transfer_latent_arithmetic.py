"""Glow style transfer with latent genre arithmetic.

This is Option B:

    z_new = z_content + alpha * (mean(z_target) - mean(z_source))

It uses a trained Glow checkpoint plus centroids computed by
compute_latent_centroids.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
GLOW_DIR = ROOT / "glow"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GLOW_DIR) not in sys.path:
    sys.path.insert(0, str(GLOW_DIR))

from glow.glow_utils import (
    AUDIO_EXTS,
    SR,
    build_model_from_checkpoint,
    choose_device,
    limit_high_frequency_growth,
    load_checkpoint,
    load_input_mel,
    mel_to_audio,
    mel_to_audio_with_input_phase,
    output_paths,
    print_mel_stats,
    smooth_mel,
)


def load_centroids(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_latent_arithmetic_on_full_mel(
    mel: np.ndarray,
    model: torch.nn.Module,
    source_label: int,
    target_label: int,
    source_centroid: torch.Tensor,
    target_centroid: torch.Tensor,
    crop_time: int,
    device: torch.device,
    hop_time: int,
    alpha: float,
) -> np.ndarray:
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
    window = np.maximum(np.hanning(crop_time).astype(np.float32), 1e-3)[None, :]

    source = torch.tensor([source_label], dtype=torch.long, device=device)
    target = torch.tensor([target_label], dtype=torch.long, device=device)
    style_vec = (target_centroid - source_centroid).to(device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for start in starts:
            chunk = mel[:, start : start + crop_time]
            original_width = chunk.shape[1]
            if original_width < crop_time:
                chunk = np.pad(chunk, ((0, 0), (0, crop_time - original_width)), mode="edge")
            x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)
            z, _ = model.encode(x, source)
            translated = model.decode(z + float(alpha) * style_vec, target)
            out = translated[0, 0].cpu().numpy().astype(np.float32)[:, :original_width]

            weight = window[:, :original_width]
            out_sum[:, start : start + original_width] += out * weight
            weight_sum[:, start : start + original_width] += weight

    return np.clip(out_sum / np.maximum(weight_sum, 1e-6), 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Glow latent-arithmetic style transfer for one audio or mel file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--direction", choices=("a2b", "b2a"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--save_mel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth_time", type=int, default=3)
    parser.add_argument("--smooth_freq", type=int, default=3)
    parser.add_argument("--limit_high_freq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high_freq_start", type=float, default=0.65)
    parser.add_argument("--high_freq_max_gain", type=float, default=1.0)
    parser.add_argument("--high_freq_margin", type=float, default=0.001)
    parser.add_argument("--hop_time", type=int, default=None)
    parser.add_argument("--assumed_max", type=float, default=1.0)
    parser.add_argument("--n_iter", "--griffin_lim_iters", type=int, default=64)
    parser.add_argument("--phase_source", choices=("input", "griffinlim"), default="input")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    centroids = load_centroids(args.centroids)
    genre_a = ckpt.get("genre_a", centroids.get("genre_a", "genre_a"))
    genre_b = ckpt.get("genre_b", centroids.get("genre_b", "genre_b"))
    source_label, target_label = (0, 1) if args.direction == "a2b" else (1, 0)
    source_centroid = centroids["centroid_a"] if args.direction == "a2b" else centroids["centroid_b"]
    target_centroid = centroids["centroid_b"] if args.direction == "a2b" else centroids["centroid_a"]
    direction_label = f"{genre_a}_to_{genre_b}_latent" if args.direction == "a2b" else f"{genre_b}_to_{genre_a}_latent"

    device = choose_device(args.device)
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    n_mels = int(ckpt["n_mels"])
    crop_time = int(ckpt["crop_time"])
    hop_time = args.hop_time if args.hop_time is not None else max(1, crop_time // 2)
    input_mel, input_mel_max = load_input_mel(args.input, n_mels, assumed_max=args.assumed_max)
    print_mel_stats("Input", input_mel)

    transferred_mel = run_latent_arithmetic_on_full_mel(
        input_mel,
        model=model,
        source_label=source_label,
        target_label=target_label,
        source_centroid=source_centroid,
        target_centroid=target_centroid,
        crop_time=crop_time,
        device=device,
        hop_time=hop_time,
        alpha=args.alpha,
    )
    print_mel_stats("Raw transferred", transferred_mel)
    if args.smooth_time > 1 or args.smooth_freq > 1:
        transferred_mel = smooth_mel(transferred_mel, time_width=args.smooth_time, freq_width=args.smooth_freq)
        print_mel_stats("Smoothed transferred", transferred_mel)
    if args.limit_high_freq:
        transferred_mel = limit_high_frequency_growth(
            transferred_mel,
            input_mel,
            start_ratio=args.high_freq_start,
            max_gain=args.high_freq_max_gain,
            margin=args.high_freq_margin,
        )
        print_mel_stats("High-frequency limited transferred", transferred_mel)

    output_wav, output_mel = output_paths(args.input, args.output, direction_label, args.save_mel)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if output_mel is not None:
        output_mel.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mel, transferred_mel)

    reconstruction_max = input_mel_max if args.input.suffix.lower() in AUDIO_EXTS else args.assumed_max
    if args.phase_source == "input" and args.input.suffix.lower() in AUDIO_EXTS:
        audio = mel_to_audio_with_input_phase(
            transferred_mel,
            input_audio_path=args.input,
            assumed_max=reconstruction_max,
            n_mels=n_mels,
        )
    else:
        audio = mel_to_audio(transferred_mel, assumed_max=reconstruction_max, n_iter=args.n_iter, mel_scale="power")
    sf.write(output_wav, audio, SR)

    print(f"Checkpoint pair: {genre_a} <-> {genre_b}")
    print(f"Direction: {direction_label}")
    print(f"Alpha: {args.alpha}")
    print(f"Reconstruction assumed_max: {reconstruction_max}")
    if output_mel is not None:
        print(f"Wrote mel: {output_mel}")
    print(f"Wrote wav: {output_wav}")


if __name__ == "__main__":
    main()
