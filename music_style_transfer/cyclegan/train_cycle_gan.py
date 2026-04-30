"""
Train a two-genre CycleGAN on *_mel_norm.npy files (shape n_mels x time, values in [0,1]).

Example:
  python cyclegan\train_cycle_gan.py --genre_a blues --genre_b jazz --epochs 100
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cycle_gan import Discriminator, Generator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEL_DIR = ROOT / "outputs" / "mel_spectrograms"


def genre_from_npy_name(name: str) -> str | None:
    """blues__blues.00000_mel_norm.npy -> blues"""
    if not name.endswith("_mel_norm.npy"):
        return None
    stem = name[: -len("_mel_norm.npy")]
    parts = stem.split("__", 1)
    return parts[0] if parts else None


class MelCropDataset(Dataset):
    """Loads random crops of shape (1, n_mels, crop_time) from disk."""

    def __init__(self, paths: list[Path], n_mels: int, crop_time: int) -> None:
        self.paths = paths
        self.n_mels = n_mels
        self.crop_time = crop_time

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        x = np.load(p).astype(np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D mel in {p}, got {x.shape}")
        h, w = x.shape
        if h != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mels in {p}, got {h}")
        if w < self.crop_time:
            pad = self.crop_time - w
            x = np.pad(x, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
            w = x.shape[1]
        start = random.randint(0, w - self.crop_time)
        crop = x[:, start : start + self.crop_time]
        t = torch.from_numpy(crop).unsqueeze(0)
        return t.clamp(0.0, 1.0)


def list_npy_for_genre(mel_dir: Path, genre: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(mel_dir.glob("*_mel_norm.npy")):
        g = genre_from_npy_name(p.name)
        if g == genre:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train two-genre CycleGAN on mel .npy tensors.")
    ap.add_argument("--genre_a", type=str, required=True)
    ap.add_argument("--genre_b", type=str, required=True)
    ap.add_argument("--mel_dir", type=Path, default=DEFAULT_MEL_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--lambda_cycle", type=float, default=10.0)
    ap.add_argument("--lambda_id", type=float, default=5.0)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--crop_time", type=int, default=128)
    ap.add_argument("--checkpoint_dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")

    paths_a = list_npy_for_genre(args.mel_dir, args.genre_a)
    paths_b = list_npy_for_genre(args.mel_dir, args.genre_b)
    if len(paths_a) == 0 or len(paths_b) == 0:
        raise SystemExit(
            f"No .npy for genres (found A={len(paths_a)}, B={len(paths_b)}). "
            f"Check --mel_dir and names like '{args.genre_a}__..._mel_norm.npy'."
        )

    ds_a = MelCropDataset(paths_a, args.n_mels, args.crop_time)
    ds_b = MelCropDataset(paths_b, args.n_mels, args.crop_time)
    loader_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    loader_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    G_AB = Generator(args.n_mels, args.crop_time).to(device)
    G_BA = Generator(args.n_mels, args.crop_time).to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    opt_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=args.lr,
        betas=(args.beta1, 0.999),
    )
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=args.lr,
        betas=(args.beta1, 0.999),
    )

    loss_gan = nn.MSELoss()
    loss_l1 = nn.L1Loss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps = min(len(loader_a), len(loader_b))
    it_a = iter(loader_a)
    it_b = iter(loader_b)

    for epoch in range(1, args.epochs + 1):
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        ep_loss_g = 0.0
        ep_loss_d = 0.0
        n_batches = 0

        for _ in range(steps):
            try:
                real_a = next(it_a)
            except StopIteration:
                it_a = iter(loader_a)
                real_a = next(it_a)
            try:
                real_b = next(it_b)
            except StopIteration:
                it_b = iter(loader_b)
                real_b = next(it_b)

            real_a = real_a.to(device)
            real_b = real_b.to(device)

            opt_G.zero_grad(set_to_none=True)
            fake_b = G_AB(real_a)
            fake_a = G_BA(real_b)
            rec_a = G_BA(fake_b)
            rec_b = G_AB(fake_a)
            id_a = G_BA(real_a)
            id_b = G_AB(real_b)

            out_db = D_B(fake_b)
            out_da = D_A(fake_a)
            loss_g_ab = loss_gan(out_db, torch.ones_like(out_db))
            loss_g_ba = loss_gan(out_da, torch.ones_like(out_da))
            loss_cycle = loss_l1(rec_a, real_a) + loss_l1(rec_b, real_b)
            loss_id = loss_l1(id_a, real_a) + loss_l1(id_b, real_b)
            loss_g = loss_g_ab + loss_g_ba + args.lambda_cycle * loss_cycle + args.lambda_id * loss_id
            loss_g.backward()
            opt_G.step()

            opt_D.zero_grad(set_to_none=True)
            loss_d_a = loss_gan(D_A(real_a), torch.ones_like(D_A(real_a)))
            loss_d_a += loss_gan(D_A(fake_a.detach()), torch.zeros_like(D_A(fake_a)))
            loss_d_a *= 0.5
            loss_d_b = loss_gan(D_B(real_b), torch.ones_like(D_B(real_b)))
            loss_d_b += loss_gan(D_B(fake_b.detach()), torch.zeros_like(D_B(fake_b)))
            loss_d_b *= 0.5
            loss_d = loss_d_a + loss_d_b
            loss_d.backward()
            opt_D.step()

            ep_loss_g += float(loss_g.detach())
            ep_loss_d += float(loss_d.detach())
            n_batches += 1

        if n_batches:
            print(
                f"epoch {epoch}/{args.epochs}  G={ep_loss_g / n_batches:.4f}  D={ep_loss_d / n_batches:.4f}"
            )

        ckpt = {
            "G_AB": G_AB.state_dict(),
            "G_BA": G_BA.state_dict(),
            "D_A": D_A.state_dict(),
            "D_B": D_B.state_dict(),
            "genre_a": args.genre_a,
            "genre_b": args.genre_b,
            "n_mels": args.n_mels,
            "crop_time": args.crop_time,
            "epoch": epoch,
        }
        torch.save(ckpt, args.checkpoint_dir / "cycle_gan.pt")

    print(f"Wrote {args.checkpoint_dir / 'cycle_gan.pt'}")


if __name__ == "__main__":
    main()
