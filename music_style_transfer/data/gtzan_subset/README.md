# GTZAN audio (local only)

Put the **genre folders** here (`blues/`, `jazz/`, …) so each track is under this directory — for example:

`gtzan_subset/blues/blues.00000.wav`

**Nothing in this project reads audio from other paths** after you copy or import once. The preprocessor only loads files under `data/gtzan_subset/`.

### First-time setup (any machine)

1. Obtain the GTZAN archive (e.g. download and unzip).
2. Run from `music_style_transfer`:

   `python import_gtzan.py "C:\path\to\folder\with\genre\subfolders"`

   That **copies** into this folder. After that, you can delete the original unzip if you want.

### GitHub

Raw **`.wav`** files are **not** committed (too large; dataset has its own terms). After cloning the repo on a new PC, download GTZAN again and run `import_gtzan.py` once, or copy this folder from a backup.
