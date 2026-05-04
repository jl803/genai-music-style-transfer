"""Plot Glow training losses from checkpoints.

Existing checkpoints may only contain the final loss, so this script creates:
1. a line plot when loss_history is available,
2. otherwise a final-loss comparison bar chart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def label_for(path: Path, ckpt: dict) -> str:
    genre_a = ckpt.get("genre_a")
    genre_b = ckpt.get("genre_b")
    epoch = ckpt.get("epoch")
    if genre_a and genre_b and epoch:
        return f"{genre_a}->{genre_b}\n{epoch} epochs"
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Glow loss history or final-loss comparison from checkpoint files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "glow_loss_plot.png")
    args = parser.parse_args()

    loaded: list[tuple[Path, dict]] = []
    for path in args.checkpoints:
        ckpt = load_checkpoint(path)
        if "model" not in ckpt:
            print(f"Skipping non-model checkpoint: {path}")
            continue
        loaded.append((path, ckpt))
    if not loaded:
        raise SystemExit("No valid Glow model checkpoints provided.")

    histories = [(path, ckpt.get("loss_history")) for path, ckpt in loaded if ckpt.get("loss_history")]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    if histories:
        for path, history in histories:
            epochs = range(1, len(history) + 1)
            plt.plot(epochs, history, label=label_for(path, load_checkpoint(path)))
        plt.xlabel("Epoch")
        plt.ylabel("NLL per dimension")
        plt.title("Glow Training Loss")
        plt.legend()
    else:
        labels = [label_for(path, ckpt) for path, ckpt in loaded]
        losses = [float(ckpt["loss"]) for _path, ckpt in loaded]
        bars = plt.bar(labels, losses)
        plt.ylabel("Final NLL per dimension")
        plt.title("Glow Final Training Loss by Checkpoint")
        plt.xticks(rotation=20, ha="right")
        for bar, loss in zip(bars, losses):
            plt.text(bar.get_x() + bar.get_width() / 2, loss, f"{loss:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    plt.close()
    print(f"Wrote loss plot: {args.output}")


if __name__ == "__main__":
    main()
