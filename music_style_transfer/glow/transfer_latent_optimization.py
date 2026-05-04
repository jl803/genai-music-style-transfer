"""Classifier-guided Glow latent optimization.

This is a stronger Glow transfer method than centroid arithmetic. For each mel
chunk, Glow encodes the source chunk to latent z, then z is optimized so the
decoded chunk is classified as the target genre while staying close to the
original content.
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

from glow.genre_classifier import load_classifier_checkpoint
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


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    time_loss = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    freq_loss = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return time_loss + freq_loss


def optimize_chunk(
    chunk: np.ndarray,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    source_label: int,
    target_label: int,
    device: torch.device,
    steps: int,
    lr: float,
    target_weight: float,
    content_weight: float,
    low_content_weight: float,
    latent_weight: float,
    high_freq_weight: float,
    smooth_weight: float,
    high_freq_start: float,
    high_freq_margin: float,
) -> tuple[np.ndarray, float]:
    x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)
    source = torch.tensor([source_label], dtype=torch.long, device=device)
    target = torch.tensor([target_label], dtype=torch.long, device=device)

    model.eval()
    classifier.eval()
    with torch.no_grad():
        z0, _ = model.encode(x, source)

    z = z0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    high_start = int(round(x.shape[2] * high_freq_start))
    low_end = max(1, high_start)

    final_prob = 0.0
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        decoded = model.decode(z, target)
        logits = classifier(decoded)
        target_loss = F.cross_entropy(logits, target)
        content_loss = F.l1_loss(decoded, x)
        low_loss = F.l1_loss(decoded[:, :, :low_end, :], x[:, :, :low_end, :])
        latent_loss = F.mse_loss(z, z0)
        high_penalty = torch.relu(decoded[:, :, high_start:, :] - x[:, :, high_start:, :] - high_freq_margin).mean()
        loss = (
            target_weight * target_loss
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
            final_prob = float(torch.softmax(logits, dim=1)[0, target_label].detach().cpu())

    with torch.no_grad():
        decoded = model.decode(z, target)[0, 0].detach().cpu().numpy().astype(np.float32)
    return np.clip(decoded, 0.0, 1.0), final_prob


def run_optimization_on_full_mel(
    mel: np.ndarray,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    source_label: int,
    target_label: int,
    crop_time: int,
    device: torch.device,
    hop_time: int,
    steps: int,
    lr: float,
    target_weight: float,
    content_weight: float,
    low_content_weight: float,
    latent_weight: float,
    high_freq_weight: float,
    smooth_weight: float,
    high_freq_start: float,
    high_freq_margin: float,
) -> tuple[np.ndarray, float]:
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
    probs: list[float] = []

    for idx, start in enumerate(starts, start=1):
        chunk = mel[:, start : start + crop_time]
        original_width = chunk.shape[1]
        if original_width < crop_time:
            chunk = np.pad(chunk, ((0, 0), (0, crop_time - original_width)), mode="edge")
        optimized, prob = optimize_chunk(
            chunk,
            model=model,
            classifier=classifier,
            source_label=source_label,
            target_label=target_label,
            device=device,
            steps=steps,
            lr=lr,
            target_weight=target_weight,
            content_weight=content_weight,
            low_content_weight=low_content_weight,
            latent_weight=latent_weight,
            high_freq_weight=high_freq_weight,
            smooth_weight=smooth_weight,
            high_freq_start=high_freq_start,
            high_freq_margin=high_freq_margin,
        )
        optimized = optimized[:, :original_width]
        weight = window[:, :original_width]
        out_sum[:, start : start + original_width] += optimized * weight
        weight_sum[:, start : start + original_width] += weight
        probs.append(prob)
        if idx == 1 or idx == len(starts) or idx % 10 == 0:
            print(f"optimized chunk {idx}/{len(starts)} target_prob={prob:.3f}")

    out = out_sum / np.maximum(weight_sum, 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32), float(np.mean(probs))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run classifier-guided Glow latent optimization for one audio or mel file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classifier_checkpoint", type=Path, required=True)
    parser.add_argument("--direction", choices=("a2b", "b2a"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--target_weight", type=float, default=3.0)
    parser.add_argument("--content_weight", type=float, default=0.5)
    parser.add_argument("--low_content_weight", type=float, default=1.0)
    parser.add_argument("--latent_weight", type=float, default=0.01)
    parser.add_argument("--high_freq_weight", type=float, default=0.2)
    parser.add_argument("--smooth_weight", type=float, default=0.02)
    parser.add_argument("--blend", type=float, default=0.75)
    parser.add_argument("--save_mel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth_time", type=int, default=3)
    parser.add_argument("--smooth_freq", type=int, default=3)
    parser.add_argument("--limit_high_freq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high_freq_start", type=float, default=0.65)
    parser.add_argument("--high_freq_max_gain", type=float, default=1.8)
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
    direction_label = f"{genre_a}_to_{genre_b}_optimized" if args.direction == "a2b" else f"{genre_b}_to_{genre_a}_optimized"

    device = choose_device(args.device)
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    classifier, classifier_info = load_classifier_checkpoint(args.classifier_checkpoint, device)

    n_mels = int(ckpt["n_mels"])
    crop_time = int(ckpt["crop_time"])
    if int(classifier_info["n_mels"]) != n_mels or int(classifier_info["crop_time"]) != crop_time:
        raise SystemExit("Classifier and Glow checkpoint must use the same n_mels and crop_time.")

    hop_time = args.hop_time if args.hop_time is not None else max(1, crop_time // 2)
    input_mel, input_mel_max = load_input_mel(args.input, n_mels, assumed_max=args.assumed_max)
    print_mel_stats("Input", input_mel)

    optimized_mel, avg_prob = run_optimization_on_full_mel(
        input_mel,
        model=model,
        classifier=classifier,
        source_label=source_label,
        target_label=target_label,
        crop_time=crop_time,
        device=device,
        hop_time=hop_time,
        steps=args.steps,
        lr=args.lr,
        target_weight=args.target_weight,
        content_weight=args.content_weight,
        low_content_weight=args.low_content_weight,
        latent_weight=args.latent_weight,
        high_freq_weight=args.high_freq_weight,
        smooth_weight=args.smooth_weight,
        high_freq_start=args.high_freq_start,
        high_freq_margin=args.high_freq_margin,
    )
    print_mel_stats("Raw optimized", optimized_mel)

    blend = float(np.clip(args.blend, 0.0, 1.0))
    optimized_mel = np.clip(input_mel * (1.0 - blend) + optimized_mel * blend, 0.0, 1.0).astype(np.float32)
    print_mel_stats("Blended optimized", optimized_mel)
    if args.smooth_time > 1 or args.smooth_freq > 1:
        optimized_mel = smooth_mel(optimized_mel, time_width=args.smooth_time, freq_width=args.smooth_freq)
        print_mel_stats("Smoothed optimized", optimized_mel)
    if args.limit_high_freq:
        optimized_mel = limit_high_frequency_growth(
            optimized_mel,
            input_mel,
            start_ratio=args.high_freq_start,
            max_gain=args.high_freq_max_gain,
            margin=args.high_freq_margin,
        )
        print_mel_stats("High-frequency limited optimized", optimized_mel)

    output_wav, output_mel = output_paths(args.input, args.output, direction_label, args.save_mel)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if output_mel is not None:
        output_mel.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mel, optimized_mel)

    reconstruction_max = input_mel_max if args.input.suffix.lower() in AUDIO_EXTS else args.assumed_max
    if args.phase_source == "input" and args.input.suffix.lower() in AUDIO_EXTS:
        audio = mel_to_audio_with_input_phase(
            optimized_mel,
            input_audio_path=args.input,
            assumed_max=reconstruction_max,
            n_mels=n_mels,
        )
    else:
        audio = mel_to_audio(optimized_mel, assumed_max=reconstruction_max, n_iter=args.n_iter, mel_scale="power")
    sf.write(output_wav, audio, SR)

    print(f"Checkpoint pair: {genre_a} <-> {genre_b}")
    print(f"Direction: {direction_label}")
    print(f"Average target classifier probability: {avg_prob:.3f}")
    print(f"Blend: {blend}")
    if output_mel is not None:
        print(f"Wrote mel: {output_mel}")
    print(f"Wrote wav: {output_wav}")


if __name__ == "__main__":
    main()
