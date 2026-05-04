"""Stem-based Glow latent-arithmetic transfer.

This keeps Glow as the generative model but applies it to the background /
instrumental stem instead of the full mixed song. The foreground stem is kept
stable by default and mixed back in after transfer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
GLOW_DIR = ROOT / "glow"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GLOW_DIR) not in sys.path:
    sys.path.insert(0, str(GLOW_DIR))

from glow.glow_utils import (
    AUDIO_EXTS,
    HOP_LENGTH,
    N_FFT,
    SR,
    build_model_from_checkpoint,
    choose_device,
    limit_high_frequency_growth,
    load_checkpoint,
    mel_to_audio,
    print_mel_stats,
    smooth_mel,
)
from glow.transfer_latent_arithmetic import load_centroids, run_latent_arithmetic_on_full_mel
from reconstruct_wav import peak_normalize


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def match_length(y: np.ndarray, length: int) -> np.ndarray:
    if len(y) == length:
        return y
    if len(y) > length:
        return y[:length]
    return np.pad(y, (0, length - len(y)))


def audio_array_to_normalized_mel(y: np.ndarray, n_mels: int) -> tuple[np.ndarray, float]:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        power=2.0,
    ).astype(np.float32)
    mel_min = float(mel.min())
    mel_max = float(mel.max())
    mel_norm = ((mel - mel_min) / (mel_max - mel_min + 1e-8)).astype(np.float32)
    return np.clip(mel_norm, 0.0, 1.0), mel_max


def mel_to_audio_with_array_phase(mel_norm: np.ndarray, phase_audio: np.ndarray, assumed_max: float) -> np.ndarray:
    mel_power = np.maximum(mel_norm * assumed_max, 0.0)
    stft_magnitude = librosa.feature.inverse.mel_to_stft(
        mel_power,
        sr=SR,
        n_fft=N_FFT,
        power=2.0,
        fmin=0.0,
        fmax=SR / 2,
    )
    phase_stft = librosa.stft(phase_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    phase = np.exp(1j * np.angle(phase_stft))
    width = min(stft_magnitude.shape[1], phase.shape[1])
    audio = librosa.istft(
        stft_magnitude[:, :width] * phase[:, :width],
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        length=len(phase_audio),
    )
    return audio.astype(np.float32)


def blend_with_original(transferred: np.ndarray, original: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    return np.clip(original * (1.0 - amount) + transferred * amount, 0.0, 1.0).astype(np.float32)


def separate_hpss(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    harmonic, percussive = librosa.effects.hpss(y)
    # HPSS is only a fallback; harmonic is the foreground-ish layer and
    # percussive/background is the safer layer to style-transfer.
    return harmonic.astype(np.float32), percussive.astype(np.float32)


def separate_demucs_direct(audio_path: Path, model_name: str, device_name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise RuntimeError("Demucs is not installed in this Python environment.") from exc

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    print(f"Loading Demucs model: {model_name} on {device}")
    model = get_model(model_name).to(device).eval()
    y, _ = librosa.load(audio_path, sr=model.samplerate, mono=True)
    wav = torch.from_numpy(np.stack([y, y], axis=0)).float()

    ref = wav.mean(0)
    wav = wav - ref.mean()
    wav = wav / ref.std().clamp_min(1e-8)
    with torch.no_grad():
        sources = apply_model(
            model,
            wav.unsqueeze(0),
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True,
        )[0].cpu()
    sources = sources * ref.std() + ref.mean()

    source_names = list(model.sources)
    vocals_index = source_names.index("vocals")
    foreground = sources[vocals_index].mean(dim=0).numpy()
    background = torch.stack([src for i, src in enumerate(sources) if i != vocals_index]).sum(dim=0).mean(dim=0).numpy()
    foreground = librosa.resample(foreground.astype(np.float32), orig_sr=model.samplerate, target_sr=SR)
    background = librosa.resample(background.astype(np.float32), orig_sr=model.samplerate, target_sr=SR)
    return foreground.astype(np.float32), background.astype(np.float32)


def separate(audio_path: Path, separator: str, demucs_model: str, device: str) -> tuple[np.ndarray, np.ndarray, str]:
    y = load_audio(audio_path)
    if separator == "demucs":
        foreground, background = separate_demucs_direct(audio_path, demucs_model, device)
        return match_length(foreground, len(y)), match_length(background, len(y)), "demucs"
    if separator == "auto":
        try:
            foreground, background = separate_demucs_direct(audio_path, demucs_model, device)
            return match_length(foreground, len(y)), match_length(background, len(y)), "demucs"
        except Exception as exc:
            print(f"Demucs unavailable ({exc}); falling back to HPSS.")
    foreground, background = separate_hpss(y)
    return foreground, background, "hpss"


def output_paths(input_path: Path, output: Path, direction_label: str, save_mel: bool) -> tuple[Path, Path | None]:
    output_is_dir = output.exists() and output.is_dir()
    if not output_is_dir and output.suffix.lower() != ".wav":
        output_is_dir = True
    if output_is_dir:
        output_wav = output / f"{input_path.stem}_{direction_label}_stem_glow.wav"
    else:
        output_wav = output
    output_mel = output_wav.with_suffix(".npy") if save_mel else None
    return output_wav, output_mel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a song into stems, run Glow latent transfer on the background stem, and recombine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--direction", choices=("a2b", "b2a"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--foreground_alpha", type=float, default=0.0)
    parser.add_argument("--transfer_foreground", action="store_true")
    parser.add_argument("--blend", type=float, default=0.65)
    parser.add_argument("--foreground_blend", type=float, default=0.5)
    parser.add_argument("--separator", choices=("auto", "demucs", "hpss"), default="auto")
    parser.add_argument("--demucs_model", type=str, default="mdx_extra_q")
    parser.add_argument("--save_mel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_stems", action="store_true")
    parser.add_argument("--smooth_time", type=int, default=3)
    parser.add_argument("--smooth_freq", type=int, default=3)
    parser.add_argument("--limit_high_freq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high_freq_start", type=float, default=0.65)
    parser.add_argument("--high_freq_max_gain", type=float, default=1.0)
    parser.add_argument("--high_freq_margin", type=float, default=0.001)
    parser.add_argument("--hop_time", type=int, default=None)
    parser.add_argument("--phase_source", choices=("stem", "griffinlim"), default="stem")
    parser.add_argument("--n_iter", "--griffin_lim_iters", type=int, default=64)
    parser.add_argument("--foreground_gain", type=float, default=1.0)
    parser.add_argument("--background_gain", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.transfer_foreground and args.foreground_alpha <= 0.0:
        args.foreground_alpha = 0.10

    if args.input.suffix.lower() not in AUDIO_EXTS:
        raise ValueError("Stem transfer expects an audio file input.")

    ckpt = load_checkpoint(args.checkpoint)
    centroids = load_centroids(args.centroids)
    genre_a = ckpt.get("genre_a", centroids.get("genre_a", "genre_a"))
    genre_b = ckpt.get("genre_b", centroids.get("genre_b", "genre_b"))
    source_label, target_label = (0, 1) if args.direction == "a2b" else (1, 0)
    source_centroid = centroids["centroid_a"] if args.direction == "a2b" else centroids["centroid_b"]
    target_centroid = centroids["centroid_b"] if args.direction == "a2b" else centroids["centroid_a"]
    direction_label = f"{genre_a}_to_{genre_b}_latent" if args.direction == "a2b" else f"{genre_b}_to_{genre_a}_latent"

    device = choose_device(args.device)
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    foreground, background, method = separate(args.input, args.separator, args.demucs_model, args.device)
    print(f"Separation: {method}")

    n_mels = int(ckpt["n_mels"])
    crop_time = int(ckpt["crop_time"])
    hop_time = args.hop_time if args.hop_time is not None else max(1, crop_time // 2)

    background_mel, background_max = audio_array_to_normalized_mel(background, n_mels)
    print_mel_stats("Background input", background_mel)
    transferred_background_mel = run_latent_arithmetic_on_full_mel(
        background_mel,
        model=model,
        source_label=source_label,
        target_label=target_label,
        source_centroid=source_centroid,
        target_centroid=target_centroid,
        crop_time=crop_time,
        device=device,
        hop_time=hop_time,
        alpha=args.alpha,
    )
    print_mel_stats("Background raw transferred", transferred_background_mel)

    if args.smooth_time > 1 or args.smooth_freq > 1:
        transferred_background_mel = smooth_mel(transferred_background_mel, args.smooth_time, args.smooth_freq)
        print_mel_stats("Background smoothed transferred", transferred_background_mel)
    if args.limit_high_freq:
        transferred_background_mel = limit_high_frequency_growth(
            transferred_background_mel,
            background_mel,
            start_ratio=args.high_freq_start,
            max_gain=args.high_freq_max_gain,
            margin=args.high_freq_margin,
        )
        print_mel_stats("Background high-frequency limited", transferred_background_mel)
    transferred_background_mel = blend_with_original(transferred_background_mel, background_mel, args.blend)
    print_mel_stats("Background blended transferred", transferred_background_mel)

    if args.phase_source == "stem":
        transferred_background = mel_to_audio_with_array_phase(transferred_background_mel, background, background_max)
    else:
        transferred_background = mel_to_audio(
            transferred_background_mel,
            assumed_max=background_max,
            n_iter=args.n_iter,
            mel_scale="power",
        )

    transferred_foreground = foreground
    if args.foreground_alpha > 0.0:
        foreground_mel, foreground_max = audio_array_to_normalized_mel(foreground, n_mels)
        transferred_foreground_mel = run_latent_arithmetic_on_full_mel(
            foreground_mel,
            model=model,
            source_label=source_label,
            target_label=target_label,
            source_centroid=source_centroid,
            target_centroid=target_centroid,
            crop_time=crop_time,
            device=device,
            hop_time=hop_time,
            alpha=args.foreground_alpha,
        )
        if args.smooth_time > 1 or args.smooth_freq > 1:
            transferred_foreground_mel = smooth_mel(transferred_foreground_mel, args.smooth_time, args.smooth_freq)
        if args.limit_high_freq:
            transferred_foreground_mel = limit_high_frequency_growth(
                transferred_foreground_mel,
                foreground_mel,
                start_ratio=args.high_freq_start,
                max_gain=args.high_freq_max_gain,
                margin=args.high_freq_margin,
            )
        transferred_foreground_mel = blend_with_original(
            transferred_foreground_mel,
            foreground_mel,
            args.foreground_blend,
        )
        transferred_foreground = mel_to_audio_with_array_phase(transferred_foreground_mel, foreground, foreground_max)

    length = max(len(transferred_foreground), len(transferred_background))
    mixed = peak_normalize(
        args.foreground_gain * match_length(transferred_foreground, length)
        + args.background_gain * match_length(transferred_background, length)
    )

    output_wav, output_mel = output_paths(args.input, args.output, direction_label, args.save_mel)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, mixed, SR)
    if output_mel is not None:
        np.save(output_mel, transferred_background_mel)
    if args.save_stems:
        stem_dir = output_wav.with_suffix("")
        stem_dir.mkdir(parents=True, exist_ok=True)
        sf.write(stem_dir / "content_foreground.wav", peak_normalize(foreground), SR)
        sf.write(stem_dir / "content_background.wav", peak_normalize(background), SR)
        sf.write(stem_dir / "transferred_foreground.wav", peak_normalize(transferred_foreground), SR)
        sf.write(stem_dir / "transferred_background.wav", peak_normalize(transferred_background), SR)
        print(f"Wrote stems: {stem_dir}")

    print(f"Checkpoint pair: {genre_a} <-> {genre_b}")
    print(f"Direction: {direction_label}")
    print(f"Background alpha: {args.alpha}")
    print(f"Background blend: {args.blend}")
    print(f"Foreground alpha: {args.foreground_alpha}")
    print(f"Foreground blend: {args.foreground_blend}")
    if output_mel is not None:
        print(f"Wrote background mel: {output_mel}")
    print(f"Wrote wav: {output_wav}")


if __name__ == "__main__":
    main()
