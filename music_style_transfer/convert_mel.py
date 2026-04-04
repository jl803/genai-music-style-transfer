"""Apply a trained two-genre checkpoint to one *_mel_norm.npy (center crop/pad to training size)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cycle_gan import Generator

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert one mel .npy with G_AB or G_BA.")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "cycle_gan.pt")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--direction",
        choices=("a2b", "b2a"),
        required=True,
        help="Map genre_a -> genre_b (a2b) or genre_b -> genre_a (b2a); must match training pair.",
    )
    args = ap.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    n_mels = int(ckpt["n_mels"])
    crop_time = int(ckpt["crop_time"])

    g = Generator(n_mels, crop_time)
    key = "G_AB" if args.direction == "a2b" else "G_BA"
    g.load_state_dict(ckpt[key])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g.to(device)
    g.eval()

    x = np.load(args.input).astype(np.float32)
    if x.shape[0] != n_mels:
        raise SystemExit(f"Expected {n_mels} mels, got {x.shape}")
    w = x.shape[1]
    if w < crop_time:
        x = np.pad(x, ((0, 0), (0, crop_time - w)), mode="constant", constant_values=0.0)
        w = x.shape[1]
    start = max(0, (w - crop_time) // 2)
    crop = x[:, start : start + crop_time]
    t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)

    with torch.no_grad():
        y = g(t)[0, 0].cpu().numpy()
    np.save(args.output, y.astype(np.float32))
    print(f"Wrote {args.output} shape={y.shape}")


if __name__ == "__main__":
    main()
