"""Train a small genre classifier on mel chunks.

The classifier is not the generator. Its job is to find the chunks that are
most strongly associated with each genre, so Glow latent centroids can be built
from cleaner style examples.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
GLOW_DIR = ROOT / "glow"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GLOW_DIR) not in sys.path:
    sys.path.insert(0, str(GLOW_DIR))

from glow.glow_utils import choose_device
from glow.train_glow import list_npy_for_genre

DEFAULT_MEL_DIR = ROOT / "outputs" / "mel_spectrograms"


class GenreChunkDataset(Dataset):
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
        if mel.ndim != 2 or mel.shape[0] != self.n_mels:
            raise ValueError(f"Expected mel shape [{self.n_mels}, T], got {mel.shape} in {path}")
        width = mel.shape[1]
        if width < self.crop_time:
            mel = np.pad(mel, ((0, 0), (0, self.crop_time - width)), mode="edge")
            width = mel.shape[1]
        start = random.randint(0, width - self.crop_time)
        crop = np.clip(mel[:, start : start + self.crop_time], 0.0, 1.0)
        return torch.from_numpy(crop).unsqueeze(0), torch.tensor(label, dtype=torch.long)


class MelGenreClassifier(nn.Module):
    def __init__(self, n_mels: int = 128, crop_time: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.crop_time = crop_time
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_items(
    paths_a: list[Path],
    paths_b: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    paths_a = paths_a.copy()
    paths_b = paths_b.copy()
    rng.shuffle(paths_a)
    rng.shuffle(paths_b)
    val_a = max(1, int(round(len(paths_a) * val_ratio))) if len(paths_a) > 1 else 0
    val_b = max(1, int(round(len(paths_b) * val_ratio))) if len(paths_b) > 1 else 0
    train = [(p, 0) for p in paths_a[val_a:]] + [(p, 1) for p in paths_b[val_b:]]
    val = [(p, 0) for p in paths_a[:val_a]] + [(p, 1) for p in paths_b[:val_b]]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_classifier_checkpoint(path: Path, device: torch.device) -> tuple[MelGenreClassifier, dict]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model = MelGenreClassifier(
        n_mels=int(ckpt["n_mels"]),
        crop_time=int(ckpt["crop_time"]),
        num_classes=2,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a mel-chunk genre classifier for filtered Glow centroids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genre_a", type=str, required=True)
    parser.add_argument("--genre_b", type=str, required=True)
    parser.add_argument("--mel_dir", type=Path, default=DEFAULT_MEL_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--crop_time", type=int, default=128)
    parser.add_argument("--chunks_per_file", type=int, default=12)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--limit_per_genre", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths_a = list_npy_for_genre(args.mel_dir, args.genre_a)
    paths_b = list_npy_for_genre(args.mel_dir, args.genre_b)
    if args.limit_per_genre is not None:
        paths_a = paths_a[: args.limit_per_genre]
        paths_b = paths_b[: args.limit_per_genre]
    if not paths_a or not paths_b:
        raise SystemExit(f"No mel files found. {args.genre_a}={len(paths_a)}, {args.genre_b}={len(paths_b)}")

    train_items, val_items = split_items(paths_a, paths_b, args.val_ratio, args.seed)
    train_ds = GenreChunkDataset(train_items, args.n_mels, args.crop_time, args.chunks_per_file)
    val_ds = GenreChunkDataset(val_items, args.n_mels, args.crop_time, max(1, args.chunks_per_file // 2))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = choose_device(args.device)
    model = MelGenreClassifier(args.n_mels, args.crop_time).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    checkpoint = args.checkpoint or ROOT / "checkpoints" / f"genre_classifier_{args.genre_a}_{args.genre_b}.pt"

    best_val_acc = -1.0
    history: list[dict[str, float]] = []
    print(f"Training classifier on {device}.")
    print(f"{args.genre_a}: {len(paths_a)} files | {args.genre_b}: {len(paths_b)} files")
    print(f"Train chunks/epoch: {len(train_ds)} | Val chunks: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_seen = 0
        for x, labels in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * x.shape[0]
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            train_seen += x.shape[0]

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(device)
                labels = labels.to(device)
                logits = model(x)
                loss = criterion(logits, labels)
                val_loss += float(loss.detach()) * x.shape[0]
                val_correct += int((logits.argmax(dim=1) == labels).sum().item())
                val_seen += x.shape[0]

        train_acc = train_correct / max(1, train_seen)
        val_acc = val_correct / max(1, val_seen)
        train_loss /= max(1, train_seen)
        val_loss /= max(1, val_seen)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "genre_a": args.genre_a,
                    "genre_b": args.genre_b,
                    "n_mels": args.n_mels,
                    "crop_time": args.crop_time,
                    "chunks_per_file": args.chunks_per_file,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                checkpoint,
            )

    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Wrote classifier checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
