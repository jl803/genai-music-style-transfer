"""Compute genre latent centroids for a trained Glow checkpoint.

This implements the first half of latent style arithmetic:

    style_vec = mean(z_target) - mean(z_source)

The companion script transfer_latent_arithmetic.py applies that vector to a
content mel before decoding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GLOW_DIR = ROOT / "glow"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GLOW_DIR) not in sys.path:
    sys.path.insert(0, str(GLOW_DIR))

from glow.glow_utils import build_model_from_checkpoint, choose_device, load_checkpoint
from glow.genre_classifier import load_classifier_checkpoint
from glow.train_glow import list_npy_for_genre

DEFAULT_MEL_DIR = ROOT / "outputs" / "mel_spectrograms"


def crop_or_pad_center(mel: np.ndarray, crop_time: int) -> np.ndarray:
    if mel.shape[1] == crop_time:
        return mel
    if mel.shape[1] > crop_time:
        start = max(0, (mel.shape[1] - crop_time) // 2)
        return mel[:, start : start + crop_time]
    return np.pad(mel, ((0, 0), (0, crop_time - mel.shape[1])), mode="edge")


def chunk_starts(width: int, crop_time: int, chunks_per_file: int) -> list[int]:
    if width <= crop_time:
        return [0]
    chunks_per_file = max(1, chunks_per_file)
    if chunks_per_file == 1:
        return [max(0, (width - crop_time) // 2)]
    return [
        int(round(pos))
        for pos in np.linspace(0, width - crop_time, num=chunks_per_file)
    ]


def crop_or_pad_at(mel: np.ndarray, crop_time: int, start: int) -> np.ndarray:
    if mel.shape[1] <= crop_time:
        return np.pad(mel, ((0, 0), (0, crop_time - mel.shape[1])), mode="edge")
    return mel[:, start : start + crop_time]


def encode_centroid(
    model: torch.nn.Module,
    paths: list[Path],
    label: int,
    n_mels: int,
    crop_time: int,
    chunks_per_file: int,
    batch_size: int,
    device: torch.device,
    classifier: torch.nn.Module | None = None,
    top_fraction: float = 1.0,
) -> torch.Tensor:
    scored_zs: list[tuple[float, torch.Tensor]] = []
    model.eval()
    if classifier is not None:
        classifier.eval()
    top_fraction = float(np.clip(top_fraction, 0.0, 1.0))
    with torch.no_grad():
        for path in paths:
            mel = np.load(path).astype(np.float32)
            if mel.ndim != 2 or mel.shape[0] != n_mels:
                raise ValueError(f"Expected mel shape [{n_mels}, T], got {mel.shape} in {path}")
            mel = np.clip(mel, 0.0, 1.0)
            crops = [
                crop_or_pad_at(mel, crop_time, start)
                for start in chunk_starts(mel.shape[1], crop_time, chunks_per_file)
            ]
            for batch_start in range(0, len(crops), batch_size):
                batch = np.stack(crops[batch_start : batch_start + batch_size], axis=0)
                x = torch.from_numpy(batch).unsqueeze(1).to(device)
                labels = torch.full((x.shape[0],), label, dtype=torch.long, device=device)
                z, _ = model.encode(x, labels)
                if classifier is None or top_fraction >= 1.0:
                    scores = torch.ones(x.shape[0], device=device)
                else:
                    probs = torch.softmax(classifier(x), dim=1)
                    scores = probs[:, label]
                for score, z_item in zip(scores.detach().cpu(), z.detach().cpu(), strict=False):
                    scored_zs.append((float(score), z_item.unsqueeze(0)))
    if not scored_zs:
        raise ValueError("No latent vectors were encoded.")
    scored_zs.sort(key=lambda item: item[0], reverse=True)
    keep = max(1, int(round(len(scored_zs) * top_fraction)))
    selected = scored_zs[:keep]
    print(
        f"label {label}: kept {keep}/{len(scored_zs)} chunks "
        f"(score range {selected[-1][0]:.3f}-{selected[0][0]:.3f})"
    )
    return torch.cat([z for _score, z in selected], dim=0).mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute source/target latent centroids for Glow latent style arithmetic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--genre_a", type=str, default=None)
    parser.add_argument("--genre_b", type=str, default=None)
    parser.add_argument("--mel_dir", type=Path, default=DEFAULT_MEL_DIR)
    parser.add_argument("--limit_per_genre", type=int, default=None)
    parser.add_argument("--chunks_per_file", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--classifier_checkpoint",
        type=Path,
        default=None,
        help="Optional genre classifier checkpoint for confidence-filtered centroids.",
    )
    parser.add_argument(
        "--top_fraction",
        type=float,
        default=1.0,
        help="Keep only this fraction of most class-confident chunks when classifier is provided.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    genre_a = args.genre_a or ckpt.get("genre_a")
    genre_b = args.genre_b or ckpt.get("genre_b")
    if not genre_a or not genre_b:
        raise SystemExit("Provide --genre_a/--genre_b or use a checkpoint that stores genre names.")

    paths_a = list_npy_for_genre(args.mel_dir, genre_a)
    paths_b = list_npy_for_genre(args.mel_dir, genre_b)
    if args.limit_per_genre is not None:
        paths_a = paths_a[: args.limit_per_genre]
        paths_b = paths_b[: args.limit_per_genre]
    if not paths_a or not paths_b:
        raise SystemExit(f"No mel files found. {genre_a}={len(paths_a)}, {genre_b}={len(paths_b)} in {args.mel_dir}")

    device = choose_device(args.device)
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    classifier = None
    classifier_info = None
    if args.classifier_checkpoint is not None:
        classifier, classifier_info = load_classifier_checkpoint(args.classifier_checkpoint, device)

    n_mels = int(ckpt["n_mels"])
    crop_time = int(ckpt["crop_time"])
    if classifier_info is not None:
        if int(classifier_info["n_mels"]) != n_mels or int(classifier_info["crop_time"]) != crop_time:
            raise SystemExit(
                "Classifier n_mels/crop_time must match the Glow checkpoint. "
                f"Classifier={classifier_info['n_mels']}x{classifier_info['crop_time']}, "
                f"Glow={n_mels}x{crop_time}."
            )
    centroid_a = encode_centroid(
        model,
        paths_a,
        0,
        n_mels,
        crop_time,
        args.chunks_per_file,
        args.batch_size,
        device,
        classifier=classifier,
        top_fraction=args.top_fraction,
    )
    centroid_b = encode_centroid(
        model,
        paths_b,
        1,
        n_mels,
        crop_time,
        args.chunks_per_file,
        args.batch_size,
        device,
        classifier=classifier,
        top_fraction=args.top_fraction,
    )

    output = args.output or args.checkpoint.with_name(f"{args.checkpoint.stem}_centroids.pt")
    torch.save(
        {
            "checkpoint": str(args.checkpoint),
            "genre_a": genre_a,
            "genre_b": genre_b,
            "n_mels": n_mels,
            "crop_time": crop_time,
            "centroid_a": centroid_a,
            "centroid_b": centroid_b,
            "count_a": len(paths_a),
            "count_b": len(paths_b),
            "chunks_per_file": args.chunks_per_file,
            "classifier_checkpoint": str(args.classifier_checkpoint) if args.classifier_checkpoint else None,
            "top_fraction": args.top_fraction,
        },
        output,
    )
    print(
        f"Computed centroids: {genre_a}={len(paths_a)} files, {genre_b}={len(paths_b)} files, "
        f"{args.chunks_per_file} chunks/file"
    )
    print(f"Wrote centroids: {output}")


if __name__ == "__main__":
    main()
