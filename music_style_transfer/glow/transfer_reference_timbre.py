"""Reference-guided Glow timbre transfer.

This method keeps Glow as the generator but uses a target reference song to
define the timbre/style. Each content chunk is encoded with Glow, then its
latent code is optimized so the decoded mel matches the reference chunk's
spectral texture while preserving the content song's low-frequency structure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

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


def spectral_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-frequency mean/std plus frequency-normalized texture."""
    mean = x.mean(dim=3)
    std = x.std(dim=3).clamp_min(1e-6)
    centered = x - x.mean(dim=(2, 3), keepdim=True)
    texture = centered / centered.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return mean, std, texture


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    batch, channels, freq, time = x.shape
    features = x.view(batch, channels * freq, time)
    return torch.bmm(features, features.transpose(1, 2)) / max(1, channels * freq * time)


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    time_loss = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    freq_loss = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return time_loss + freq_loss


def crop_or_pad(mel: np.ndarray, start: int, crop_time: int) -> np.ndarray:
    chunk = mel[:, start : start + crop_time]
    if chunk.shape[1] < crop_time:
        chunk = np.pad(chunk, ((0, 0), (0, crop_time - chunk.shape[1])), mode="edge")
    return chunk.astype(np.float32)


def make_starts(width: int, crop_time: int, hop_time: int) -> list[int]:
    starts = list(range(0, max(1, width - crop_time + 1), hop_time))
    if not starts or starts[-1] != max(0, width - crop_time):
        starts.append(max(0, width - crop_time))
    return starts


def style_start_for_index(style_width: int, crop_time: int, idx: int, total: int) -> int:
    if style_width <= crop_time:
        return 0
    if total <= 1:
        return max(0, (style_width - crop_time) // 2)
    return int(round((style_width - crop_time) * (idx / (total - 1))))


def optimize_chunk(
    content_chunk: np.ndarray,
    style_chunk: np.ndarray,
    model: torch.nn.Module,
    source_label: int,
    target_label: int,
    device: torch.device,
    steps: int,
    lr: float,
    style_weight: float,
    gram_weight: float,
    content_weight: float,
    low_content_weight: float,
    latent_weight: float,
    high_freq_weight: float,
    smooth_weight: float,
    high_freq_start: float,
    high_freq_margin: float,
) -> np.ndarray:
    content = torch.from_numpy(content_chunk).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)
    style = torch.from_numpy(style_chunk).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)
    source = torch.tensor([source_label], dtype=torch.long, device=device)
    target = torch.tensor([target_label], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        z0, _ = model.encode(content, source)
        style_mean, style_std, style_texture = spectral_stats(style)
        style_gram = gram_matrix(style_texture)

    z = z0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    high_start = int(round(content.shape[2] * high_freq_start))
    low_end = max(1, high_start)

    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        decoded = model.decode(z, target)
        decoded_mean, decoded_std, decoded_texture = spectral_stats(decoded)
        style_loss = F.l1_loss(decoded_mean, style_mean) + F.l1_loss(decoded_std, style_std)
        gram_loss = F.l1_loss(gram_matrix(decoded_texture), style_gram)
        content_loss = F.l1_loss(decoded, content)
        low_loss = F.l1_loss(decoded[:, :, :low_end, :], content[:, :, :low_end, :])
        latent_loss = F.mse_loss(z, z0)
        high_penalty = torch.relu(decoded[:, :, high_start:, :] - content[:, :, high_start:, :] - high_freq_margin).mean()
        loss = (
            style_weight * style_loss
            + gram_weight * gram_loss
            + content_weight * content_loss
            + low_content_weight * low_loss
            + latent_weight * latent_loss
            + high_freq_weight * high_penalty
            + smooth_weight * smoothness_loss(decoded)
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z.clamp_(-6.0, 6.0)

    with torch.no_grad():
        decoded = model.decode(z, target)[0, 0].detach().cpu().numpy().astype(np.float32)
    return np.clip(decoded, 0.0, 1.0)


def run_reference_timbre_transfer(
    content_mel: np.ndarray,
    style_mel: np.ndarray,
    model: torch.nn.Module,
    source_label: int,
    target_label: int,
    crop_time: int,
    hop_time: int,
    device: torch.device,
    steps: int,
    lr: float,
    style_weight: float,
    gram_weight: float,
    content_weight: float,
    low_content_weight: float,
    latent_weight: float,
    high_freq_weight: float,
    smooth_weight: float,
    high_freq_start: float,
    high_freq_margin: float,
) -> np.ndarray:
    width = content_mel.shape[1]
    starts = make_starts(width, crop_time, hop_time)
    out_sum = np.zeros((content_mel.shape[0], width), dtype=np.float32)
    weight_sum = np.zeros((1, width), dtype=np.float32)
    window = np.maximum(np.hanning(crop_time).astype(np.float32), 1e-3)[None, :]

    for idx, start in enumerate(starts):
        original_width = min(crop_time, width - start)
        content_chunk = crop_or_pad(content_mel, start, crop_time)
        style_start = style_start_for_index(style_mel.shape[1], crop_time, idx, len(starts))
        style_chunk = crop_or_pad(style_mel, style_start, crop_time)
        transferred = optimize_chunk(
            content_chunk,
            style_chunk,
            model=model,
            source_label=source_label,
            target_label=target_label,
            device=device,
            steps=steps,
            lr=lr,
            style_weight=style_weight,
            gram_weight=gram_weight,
            content_weight=content_weight,
            low_content_weight=low_content_weight,
            latent_weight=latent_weight,
            high_freq_weight=high_freq_weight,
            smooth_weight=smooth_weight,
            high_freq_start=high_freq_start,
            high_freq_margin=high_freq_margin,
        )[:, :original_width]
        weight = window[:, :original_width]
        out_sum[:, start : start + original_width] += transferred * weight
        weight_sum[:, start : start + original_width] += weight
        if idx == 0 or idx == len(starts) - 1 or (idx + 1) % 10 == 0:
            print(f"optimized chunk {idx + 1}/{len(starts)}")

    return np.clip(out_sum / np.maximum(weight_sum, 1e-6), 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reference-guided Glow timbre transfer for one audio or mel file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--direction", choices=("a2b", "b2a"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--style_weight", type=float, default=2.0)
    parser.add_argument("--gram_weight", type=float, default=0.2)
    parser.add_argument("--content_weight", type=float, default=2.0)
    parser.add_argument("--low_content_weight", type=float, default=5.0)
    parser.add_argument("--latent_weight", type=float, default=0.02)
    parser.add_argument("--high_freq_weight", type=float, default=3.0)
    parser.add_argument("--smooth_weight", type=float, default=0.25)
    parser.add_argument("--blend", type=float, default=0.45)
    parser.add_argument("--save_mel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth_time", type=int, default=3)
    parser.add_argument("--smooth_freq", type=int, default=3)
    parser.add_argument("--limit_high_freq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high_freq_start", type=float, default=0.65)
    parser.add_argument("--high_freq_max_gain", type=float, default=1.1)
    parser.add_argument("--high_freq_margin", type=float, default=0.001)
    parser.add_argument("--hop_time", type=int, default=None)
    parser.add_argument("--assumed_max", type=float, default=1.0)
    parser.add_argument("--n_iter", "--griffin_lim_iters", type=int, default=64)
    parser.add_argument("--phase_source", choices=("input", "griffinlim"), default="input")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    genre_a = ckpt.get("genre_a", "genre_a")
    genre_b = ckpt.get("genre_b", "genre_b")
    source_label, target_label = (0, 1) if args.direction == "a2b" else (1, 0)
    direction_label = (
        f"{genre_a}_to_{genre_b}_reference_timbre"
        if args.direction == "a2b"
        else f"{genre_b}_to_{genre_a}_reference_timbre"
    )

    device = choose_device(args.device)
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    n_mels = int(ckpt["n_mels"])
    crop_time = int(ckpt["crop_time"])
    hop_time = args.hop_time if args.hop_time is not None else max(1, crop_time // 2)
    input_mel, input_mel_max = load_input_mel(args.input, n_mels, assumed_max=args.assumed_max)
    style_mel, _style_mel_max = load_input_mel(args.style, n_mels, assumed_max=args.assumed_max)
    print_mel_stats("Input", input_mel)
    print_mel_stats("Style", style_mel)

    transferred_mel = run_reference_timbre_transfer(
        content_mel=input_mel,
        style_mel=style_mel,
        model=model,
        source_label=source_label,
        target_label=target_label,
        crop_time=crop_time,
        hop_time=hop_time,
        device=device,
        steps=args.steps,
        lr=args.lr,
        style_weight=args.style_weight,
        gram_weight=args.gram_weight,
        content_weight=args.content_weight,
        low_content_weight=args.low_content_weight,
        latent_weight=args.latent_weight,
        high_freq_weight=args.high_freq_weight,
        smooth_weight=args.smooth_weight,
        high_freq_start=args.high_freq_start,
        high_freq_margin=args.high_freq_margin,
    )
    print_mel_stats("Raw reference-timbre", transferred_mel)

    blend = float(np.clip(args.blend, 0.0, 1.0))
    transferred_mel = np.clip(input_mel * (1.0 - blend) + transferred_mel * blend, 0.0, 1.0).astype(np.float32)
    print_mel_stats("Blended reference-timbre", transferred_mel)
    if args.smooth_time > 1 or args.smooth_freq > 1:
        transferred_mel = smooth_mel(transferred_mel, time_width=args.smooth_time, freq_width=args.smooth_freq)
        print_mel_stats("Smoothed reference-timbre", transferred_mel)
    if args.limit_high_freq:
        transferred_mel = limit_high_frequency_growth(
            transferred_mel,
            input_mel,
            start_ratio=args.high_freq_start,
            max_gain=args.high_freq_max_gain,
            margin=args.high_freq_margin,
        )
        print_mel_stats("High-frequency limited reference-timbre", transferred_mel)

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
    print(f"Style reference: {args.style}")
    print(f"Blend: {blend}")
    if output_mel is not None:
        print(f"Wrote mel: {output_mel}")
    print(f"Wrote wav: {output_wav}")


if __name__ == "__main__":
    main()
