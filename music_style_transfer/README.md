# Script Usage

## `transfer_song.py`

Auto-name the output inside a folder:

```powershell
python transfer_song.py --input "C:\path\to\song.wav" --direction a2b --output outputs
```

Choose the exact output wav path:

```powershell
python transfer_song.py --input "C:\path\to\song.wav" --direction a2b --output outputs\my_result.wav
```

Optional:

- `--checkpoint checkpoints\cycle_gan_blues_jazz.pt`
- `--save_mel`
- `--hop_time 32`
- `--assumed_max 1 --n_iter 64`
- `--mel_scale power`

## `batch_transfer_songs.py`

```powershell
python batch_transfer_songs.py --input "C:\path\to\folder" --direction a2b --output outputs\batch
```

Optional:

- `--checkpoint checkpoints\cycle_gan_blues_jazz.pt`
- `--save_mel`
- `--hop_time 32`
- `--assumed_max 1 --n_iter 64`
- `--mel_scale power`

## `reconstruct_wav.py`

```powershell
python reconstruct_wav.py outputs\mel_spectrograms\blues__blues.00000_mel_norm.npy outputs\blues_preview.wav
```

Optional:

- `--assumed_max 1`
- `--mel_scale power`
- `--n_iter 64`

## Classifier-guided transfer experiment

Train a blues-vs-jazz classifier on the existing mel `.npy` files:

```powershell
python train_genre_classifier.py --genre_a blues --genre_b jazz --epochs 50
```

Use that classifier to nudge a song toward jazz while keeping it close to the original:

```powershell
python classifier_guided_transfer.py --input "C:\path\to\blues_song.wav" --target jazz --output outputs
```

Optional:

- `--classifier checkpoints\genre_classifier_blues_jazz.pt`
- `--save_mel`
- `--steps 200`
- `--content_weight 80 --smooth_weight 20`
- `--max_delta 0.025 --post_smooth 5`
