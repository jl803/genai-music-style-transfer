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
python glow\train_glow.py --genre_a blues --genre_b jazz --epochs 200 --batch_size 8 --checkpoint checkpoints\glow_blues_jazz.pt
```

Run blues-to-jazz transfer:

```powershell
python glow\transfer_glow.py --input "C:\path\to\blues_song.wav" --checkpoint checkpoints\glow_blues_jazz.pt --direction a2b --output outputs --save_mel
```

Use `--direction b2a` for jazz-to-blues if the checkpoint was trained with `--genre_a blues --genre_b jazz`.
