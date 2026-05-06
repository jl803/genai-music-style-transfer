# Music Style Transfer

Run commands from the `music_style_transfer` folder.

## Preprocess

Convert GTZAN WAV files into normalized mel `.npy` files:

```powershell
python preprocess_mel.py
```

Reconstruct a mel `.npy` file back to WAV:

```powershell
python reconstruct_wav.py outputs\mel_spectrograms\blues__blues.00000_mel_norm.npy outputs\blues_preview.wav
```

## CycleGAN

Train:

```powershell
python cyclegan\train_cycle_gan.py --genre_a blues --genre_b jazz --epochs 100
```

Transfer one song:

```powershell
python cyclegan\transfer_song.py --input "C:\path\to\song.wav" --direction a2b --output outputs --save_mel
```

Batch transfer a folder:

```powershell
python cyclegan\batch_transfer_songs.py --input "C:\path\to\input_folder" --direction a2b --output outputs
```

Use `--direction a2b` for `genre_a -> genre_b`, or `--direction b2a` for `genre_b -> genre_a`.

## Glow

Train Glow:

```powershell
python glow\train_glow.py --genre_a blues --genre_b metal --epochs 100 --batch_size 8 --chunks_per_file 6 --checkpoint checkpoints\glow_blues_metal_chunk.pt
```

Train the genre classifier used by guided optimization:

```powershell
python glow\genre_classifier.py --genre_a blues --genre_b metal --epochs 40 --batch_size 16 --chunks_per_file 6 --checkpoint checkpoints\genre_classifier_blues_metal.pt
```

Run latent arithmetic:

```powershell
python glow\compute_latent_centroids.py --checkpoint checkpoints\glow_blues_metal_chunk.pt --chunks_per_file 6 --output checkpoints\glow_blues_metal_centroids.pt
python glow\transfer_latent_arithmetic.py --input "C:\path\to\song.wav" --checkpoint checkpoints\glow_blues_metal_chunk.pt --centroids checkpoints\glow_blues_metal_centroids.pt --direction a2b --output outputs
```

Run stem-based latent arithmetic:

```powershell
python glow\stem_latent_transfer.py --input "C:\path\to\song.wav" --checkpoint checkpoints\glow_blues_metal_chunk.pt --centroids checkpoints\glow_blues_metal_centroids.pt --direction a2b --output outputs --transfer_foreground --save_stems
```

Run classifier-guided latent optimization:

```powershell
python glow\transfer_latent_optimization.py --input "C:\path\to\song.wav" --checkpoint checkpoints\glow_blues_metal_chunk.pt --classifier_checkpoint checkpoints\genre_classifier_blues_metal.pt --direction a2b --output outputs
```

Optional loss plot:

```powershell
python glow\plot_losses.py checkpoints\glow_blues_metal_chunk.pt checkpoints\glow_blues_classical_chunk.pt --output outputs\glow_loss_plot.png
```
