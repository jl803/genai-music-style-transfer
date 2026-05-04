# Music Style Transfer

Run commands from the `music_style_transfer` folder.

## Preprocess Audio

Convert the GTZAN audio files into normalized mel `.npy` files:

```powershell
python preprocess_mel.py
```

## Reconstruct WAV

Convert one mel `.npy` file back into a WAV file:

```powershell
python reconstruct_wav.py outputs\mel_spectrograms\blues__blues.00000_mel_norm.npy outputs\blues_preview.wav
```

## CycleGAN

Train CycleGAN:

```powershell
python cyclegan\train_cycle_gan.py --genre_a blues --genre_b jazz --epochs 100
```

Run blues-to-jazz transfer:

```powershell
python cyclegan\transfer_song.py --input "C:\path\to\blues_song.wav" --direction a2b --output outputs --save_mel
```

## Glow

Train Glow:

```powershell
python glow\train_glow.py --genre_a blues --genre_b jazz --epochs 200 --batch_size 8 --chunks_per_file 12 --checkpoint checkpoints\glow_blues_jazz.pt
```

Compute latent centroids and run latent-style arithmetic:

```powershell
python glow\compute_latent_centroids.py --checkpoint checkpoints\glow_blues_jazz.pt --chunks_per_file 12 --output checkpoints\glow_blues_jazz_centroids.pt
python glow\transfer_latent_arithmetic.py --input "C:\path\to\blues_song.wav" --checkpoint checkpoints\glow_blues_jazz.pt --centroids checkpoints\glow_blues_jazz_centroids.pt --direction a2b --output outputs
```

Use `--direction b2a` for jazz-to-blues if the checkpoint was trained with `--genre_a blues --genre_b jazz`.

For a stronger target-genre direction, train a classifier and use classifier-filtered centroids:

```powershell
python glow\genre_classifier.py --genre_a blues --genre_b jazz --epochs 40 --batch_size 16 --chunks_per_file 12 --checkpoint checkpoints\genre_classifier_blues_jazz.pt
python glow\compute_latent_centroids.py --checkpoint checkpoints\glow_blues_jazz.pt --classifier_checkpoint checkpoints\genre_classifier_blues_jazz.pt --top_fraction 0.30 --chunks_per_file 12 --output checkpoints\glow_blues_jazz_filtered_centroids.pt
```

Run stem-based Glow transfer:

```powershell
python glow\stem_latent_transfer.py --input "C:\path\to\blues_song.wav" --checkpoint checkpoints\glow_blues_jazz.pt --centroids checkpoints\glow_blues_jazz_filtered_centroids.pt --direction a2b --output outputs --transfer_foreground --save_stems
```

This separates the input into foreground/background stems, applies Glow latent transfer to the background stem, lightly transfers the foreground/vocal stem, and recombines the audio. By default it uses Demucs if available and falls back to HPSS if Demucs is not installed.

Run classifier-guided Glow latent optimization:

```powershell
python glow\transfer_latent_optimization.py --input "C:\path\to\blues_song.wav" --checkpoint checkpoints\glow_blues_jazz.pt --classifier_checkpoint checkpoints\genre_classifier_blues_jazz.pt --direction a2b --output outputs
```

This optimizes each Glow latent chunk toward the target genre classifier while keeping the generated mel close to the original song.
