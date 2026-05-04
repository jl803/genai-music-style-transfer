"""Train a conditional Glow-style flow on two genres of mel spectrograms.

Example:
    python glow\train_glow.py --genre_a blues --genre_b jazz --epochs 100
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from glow_model import ConditionalGlow, glow_nll_per_sample

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEL_DIR = ROOT / "outputs" / "mel_spectrograms"


def genre_from_npy_name(name: str) -> str | None:
    """blues__blues.00000_mel_norm.npy -> blues"""
    if not name.endswith("_mel_norm.npy"):
        return None
    stem = name[: -len("_mel_norm.npy")]
    parts = stem.split("__", 1)
    return parts[0] if parts else None


def list_npy_for_genre(mel_dir: Path, genre: str) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(mel_dir.glob("*_mel_norm.npy")):
        if genre_from_npy_name(path.name) == genre:
            paths.append(path)
    return paths


class LabeledMelCropDataset(Dataset):
    """Loads many random crops of shape [1, n_mels, crop_time] with genre labels."""

    def __init__(
        self,
        items: list[tuple[Path, int]],
        n_mels: int,
        crop_time: int,
        chunks_per_file: int,
    ) -> None:
        self.items = items
        self.n_mels = n_mels
        self.crop_time = crop_time
        self.chunks_per_file = max(1, chunks_per_file)

    def __len__(self) -> int:
        return len(self.items) * self.chunks_per_file

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[idx % len(self.items)]
        mel = np.load(path).astype(np.float32)
        if mel.ndim != 2:
            raise ValueError(f"Expected 2-D mel in {path}, got {mel.shape}.")
        if mel.shape[0] != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mel bins in {path}, got {mel.shape[0]}.")

        width = mel.shape[1]
        if width < self.crop_time:
            mel = np.pad(mel, ((0, 0), (0, self.crop_time - width)), mode="constant")
            width = mel.shape[1]

        start = random.randint(0, width - self.crop_time)
        crop = mel[:, start : start + self.crop_time]
        x = torch.from_numpy(crop).unsqueeze(0).clamp(0.0, 1.0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def save_checkpoint_safely(payload: dict, checkpoint: Path) -> Path:
    """Save through a temp file so Windows does not leave partial checkpoints."""
    tmp_path = checkpoint.with_name(f"{checkpoint.stem}.tmp.{os.getpid()}{checkpoint.suffix}")
    torch.save(payload, tmp_path)
    try:
        tmp_path.replace(checkpoint)
        return checkpoint
    except OSError as exc:
        epoch = payload.get("epoch", "latest")
        fallback = checkpoint.with_name(f"{checkpoint.stem}_epoch{epoch}{checkpoint.suffix}")
        tmp_path.replace(fallback)
        print(f"Could not replace {checkpoint} ({exc}). Saved fallback checkpoint: {fallback}")
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train conditional Glow on two genres of normalized mel .npy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genre_a", type=str, required=True, help="Source genre A, class 0.")
    parser.add_argument("--genre_b", type=str, required=True, help="Source genre B, class 1.")
    parser.add_argument("--mel_dir", type=Path, default=DEFAULT_MEL_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--crop_time", type=int, default=128)
    parser.add_argument(
        "--chunks_per_file",
        type=int,
        default=12,
        help="Random mel chunks sampled from each song per epoch.",
    )
    parser.add_argument("--limit_per_genre", type=int, default=None, help="Optional quick-test cap per genre.")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Optional quick-test batch cap per epoch.")
    parser.add_argument("--n_flows", type=int, default=12)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--cond_channels", type=int, default=32)
    parser.add_argument("--contrast_weight", type=float, default=1.0)
    parser.add_argument("--contrast_margin", type=float, default=0.5)
    parser.add_argument(
        "--classifier_checkpoint",
        type=Path,
        default=None,
        help="Optional pre-trained genre classifier checkpoint. When provided, adds a "
             "translation loss: translate x to the other genre and penalize misclassification.",
    )
    parser.add_argument(
        "--classifier_weight",
        type=float,
        default=1.0,
        help="Weight for the classifier-guided translation loss.",
    )
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Output checkpoint path. Defaults to checkpoints/glow_<genre_a>_<genre_b>.pt.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths_a = list_npy_for_genre(args.mel_dir, args.genre_a)
    paths_b = list_npy_for_genre(args.mel_dir, args.genre_b)
    if not paths_a or not paths_b:
        raise SystemExit(
            f"No .npy files found for one or both genres. "
            f"Found {args.genre_a}={len(paths_a)}, {args.genre_b}={len(paths_b)} in {args.mel_dir}."
        )
    if args.limit_per_genre is not None:
        paths_a = paths_a[: args.limit_per_genre]
        paths_b = paths_b[: args.limit_per_genre]

    items = [(path, 0) for path in paths_a] + [(path, 1) for path in paths_b]
    dataset = LabeledMelCropDataset(items, args.n_mels, args.crop_time, args.chunks_per_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    device = choose_device(args.device)
    model = ConditionalGlow(
        n_mels=args.n_mels,
        time_len=args.crop_time,
        n_flows=args.n_flows,
        hidden_channels=args.hidden_channels,
        cond_channels=args.cond_channels,
        num_classes=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    classifier = None
    if args.classifier_checkpoint is not None:
        from genre_classifier import load_classifier_checkpoint
        classifier, _ = load_classifier_checkpoint(args.classifier_checkpoint, device)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad_(False)
        print(f"Loaded classifier: {args.classifier_checkpoint}")

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = ROOT / "checkpoints" / f"glow_{args.genre_a}_{args.genre_b}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training Glow on {device}.")
    print(f"{args.genre_a}: {len(paths_a)} files | {args.genre_b}: {len(paths_b)} files")
    print(f"Training chunks per epoch: {len(dataset)} ({args.chunks_per_file} chunks/file)")
    print(f"Checkpoint: {checkpoint}")

    loss_history: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_contrast = 0.0
        total_translation = 0.0
        batches = 0

        for batch_idx, (x, labels) in enumerate(loader, start=1):
            x = x.to(device)
            labels = labels.to(device)
            wrong_labels = 1 - labels

            optimizer.zero_grad(set_to_none=True)
            correct_nll = glow_nll_per_sample(model, x, labels)
            wrong_nll = glow_nll_per_sample(model, x, wrong_labels)
            contrast_loss = F.softplus(args.contrast_margin + correct_nll - wrong_nll).mean()

            # Classifier-guided translation loss: translate x to the other genre and
            # penalize when the frozen classifier doesn't recognise the output as target.
            # This directly rewards the model for producing convincing translations.
            if classifier is not None:
                with torch.no_grad():
                    z_src, _ = model.encode(x, labels)
                x_transfer = model.decode(z_src, wrong_labels)
                translation_loss = F.cross_entropy(classifier(x_transfer), wrong_labels)
            else:
                translation_loss = x.new_zeros(1).squeeze()

            loss = (
                correct_nll.mean()
                + args.contrast_weight * contrast_loss
                + args.classifier_weight * translation_loss
            )
            if not torch.isfinite(loss):
                print("Skipping non-finite loss batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += float(loss.detach())
            total_nll += float(correct_nll.mean().detach())
            total_contrast += float(contrast_loss.detach())
            total_translation += float(translation_loss.detach())
            batches += 1
            if args.steps_per_epoch is not None and batch_idx >= args.steps_per_epoch:
                break

        avg_loss = total_loss / max(1, batches)
        avg_nll = total_nll / max(1, batches)
        avg_contrast = total_contrast / max(1, batches)
        avg_translation = total_translation / max(1, batches)
        loss_history.append(avg_loss)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"loss={avg_loss:.4f}  nll={avg_nll:.4f}  "
            f"contrast={avg_contrast:.4f}  translation={avg_translation:.4f}"
        )

        save_checkpoint_safely(
            {
                "model": model.state_dict(),
                "genre_a": args.genre_a,
                "genre_b": args.genre_b,
                "n_mels": args.n_mels,
                "crop_time": args.crop_time,
                "chunks_per_file": args.chunks_per_file,
                "n_flows": args.n_flows,
                "hidden_channels": args.hidden_channels,
                "cond_channels": args.cond_channels,
                "contrast_weight": args.contrast_weight,
                "contrast_margin": args.contrast_margin,
                "classifier_weight": args.classifier_weight,
                "epoch": epoch,
                "loss": avg_loss,
                "loss_history": loss_history,
            },
            checkpoint,
        )

    print(f"Wrote Glow checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
