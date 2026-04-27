"""Batch end-to-end music style transfer for a directory of audio files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from cycle_gan import Generator
from transfer_song import (
    ROOT,
    audio_to_normalized_mel,
    mel_to_audio,
    run_generator_on_full_mel,
)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end style transfer for all audio files in a directory.")
    ap.add_argument(
        "--input",
        "--input_dir",
        dest="input_dir",
        type=Path,
        required=True,
        help="Directory containing source audio files.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "cycle_gan.pt",
        help="Path to a checkpoint from train_cycle_gan.py",
    )
    ap.add_argument(
        "--direction",
        choices=("a2b", "b2a"),
        required=True,
        help="Use the generator that maps genre_a -> genre_b or genre_b -> genre_a.",
    )
    ap.add_argument(
        "--output",
        "--output_dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory where output wav files will be written.",
    )
    ap.add_argument("--griffin_lim_iters", type=int, default=32)
    ap.add_argument("--energy_scale", type=float, default=1000.0)
    ap.add_argument("--power_exponent", type=float, default=2.0)
    ap.add_argument("--save_mel", action="store_true", help="Also save transferred mel .npy files.")
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU.")
        device = torch.device("cpu")

    generator = Generator(n_mels, crop_time)
    key = "G_AB" if args.direction == "a2b" else "G_BA"
    generator.load_state_dict(ckpt[key])
    generator.to(device)
    generator.eval()

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
            transferred_mel = run_generator_on_full_mel(input_mel, generator, crop_time=crop_time, device=device)
            transferred_audio = mel_to_audio(
                transferred_mel,
                griffin_lim_iters=args.griffin_lim_iters,
                energy_scale=args.energy_scale,
                power_exponent=args.power_exponent,
            )

            sf.write(out_wav, transferred_audio, 22050)
            if args.save_mel:
                out_mel = out_wav.with_suffix(".npy")
                np.save(out_mel, transferred_mel.astype(np.float32))

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
