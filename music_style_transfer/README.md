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
- `--assumed_max 100 --n_iter 64`

## `batch_transfer_songs.py`

```powershell
python batch_transfer_songs.py --input "C:\path\to\folder" --direction a2b --output outputs\batch
```

Optional:

- `--checkpoint checkpoints\cycle_gan_blues_jazz.pt`
- `--save_mel`
- `--hop_time 32`
- `--assumed_max 100 --n_iter 64`

## `reconstruct_wav.py`

```powershell
python reconstruct_wav.py outputs\mel_spectrograms\blues__blues.00000_mel_norm.npy outputs\blues_preview.wav
```

Optional:

- `--assumed_max 100`
- `--n_iter 64`
