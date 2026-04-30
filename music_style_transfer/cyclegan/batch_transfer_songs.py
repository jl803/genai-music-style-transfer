"""Batch end-to-end music style transfer for a directory of audio files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from cycle_gan import Generator
from transfer_song import ROOT, audio_to_normalized_mel, run_generator_on_full_mel
from reconstruct_wav import mel_to_audio

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CycleGAN style transfer for all audio files in a directory.")
    ap.add_argument("--input", "--input_dir", dest="input_dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "cycle_gan.pt")
    ap.add_argument("--direction", choices=("a2b", "b2a"), required=True)
    ap.add_argument("--output", "--output_dir", dest="output_dir", type=Path, required=True)
    ap.add_argument("--assumed_max", type=float, default=1.0)
    ap.add_argument("--mel_scale", choices=("power", "db"), default="power")
    ap.add_argument("--n_iter", "--griffin_lim_iters", type=int, default=64)
    ap.add_argument("--hop_time", type=int, default=None)
    ap.add_argument("--save_mel", action="store_true")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    n_mels = int(ckpt["n_mels"])
    crop_time = int(ckpt["crop_time"])
    genre_a = ckpt.get("genre_a", "genre_a")
    genre_b = ckpt.get("genre_b", "genre_b")
    direction_label = f"{genre_a}_to_{genre_b}" if args.direction == "a2b" else f"{genre_b}_to_{genre_a}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA not available, using CPU.")

    generator = Generator(n_mels, crop_time)
    key = "G_AB" if args.direction == "a2b" else "G_BA"
    generator.load_state_dict(ckpt[key])
    generator.to(device)
    generator.eval()
    hop_time = args.hop_time if args.hop_time is not None else max(1, crop_time // 2)

    audio_paths = [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if not audio_paths:
        raise SystemExit(f"No supported audio files found under {input_dir}")

    converted = 0
    failed = 0

    for audio_path in audio_paths:
        rel = audio_path.relative_to(input_dir)
        base_name = f"{audio_path.stem}_{direction_label}"
        out_wav = (output_dir / rel).with_name(base_name + ".wav")
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        try:
            input_mel = audio_to_normalized_mel(audio_path, n_mels=n_mels)
            transferred_mel = run_generator_on_full_mel(
                input_mel,
                generator,
                crop_time=crop_time,
                device=device,
                hop_time=hop_time,
            )
            transferred_audio = mel_to_audio(
                transferred_mel,
                assumed_max=args.assumed_max,
                n_iter=args.n_iter,
                mel_scale=args.mel_scale,
            )

            sf.write(out_wav, transferred_audio, 22050)
            if args.save_mel:
                np.save(out_wav.with_suffix(".npy"), transferred_mel.astype(np.float32))

            converted += 1
            if converted == 1 or converted % 10 == 0:
                print(f"[{converted}] Wrote {out_wav}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {audio_path}: {exc}")

    print(f"Checkpoint pair: {genre_a} <-> {genre_b}")
    print(f"Direction: {direction_label}")
    print(f"Done. Converted {converted} file(s) to {output_dir}")
    if failed:
        print(f"Failed on {failed} file(s)")


if __name__ == "__main__":
    main()
