# LESS-WSI preprocessing (urine cytology cohort)

This folder ships the preprocessing pipeline we used for the in-house
urine cytology dataset. The exact parameters here (DeepZoom level,
intensity thresholds, the specific sliding-window screening rule) were
tuned on that private cohort. They are **not** required by the rest of
the LESS pipeline. The only invariants the method actually needs are:

- **100 informative patches per slide** (random sampling is fine), and
- a **256×256 patch** plus the **central 128×128 crop** of that same
  patch as the two scales fed into Step 0.

On a new dataset you have three reasonable options for getting there:

- (a) reuse [CLAM's](https://github.com/mahmoodlab/CLAM) WSI segmentation
  + patching with whatever foreground threshold suits your stain, then
  randomly subsample to 100 / slide;
- (b) design your own screening heuristic;
- (c) reuse the scripts below as a starting point and just retune
  `--mean_thr / --std_thr / --center_mean_thr / --dz_level_offset` for
  your scanner and stain.

The sample-and-relayout step (`subsample_and_layout.py`) is
dataset-agnostic and can sit on top of any of the three.

> Both scripts are intended to be run on the WSI host where the `.svs`
> files live; the rest of this repo (Step 0 + Step 1) only needs the
> final patch tree.

## Pipeline overview

Two stages:

1. **`sliding_window.py`** — sliding-window patch extraction with a
   white-background filter (`mean<236 ∧ std>10 ∧ center_mean<236`).
   Run **once per scale** (256 px, then 128 px).
2. **`subsample_and_layout.py`** — random 100-patch subsample per slide,
   slide-level class assignment from a CSV, and stratified 5-fold split
   shared across scales. Produces the on-disk tree that Step 0 expects.

## Step P1 — Sliding window (one pass per scale)

```bash
cd 0-feature_extraction

# 256 px (paper default)
python sliding_window.py --wsi_path /path/to/cohort --tile_size 256
# writes /path/to/cohort/save_patches_256/<slide_id>/<col>_<row>_m.png

# 128 px (same screening rule, center crop scales to 64x64)
python sliding_window.py --wsi_path /path/to/cohort --tile_size 128
# writes /path/to/cohort/save_patches_128/<slide_id>/<col>_<row>_m.png
```

Per-scale outputs land in `<wsi_path>/save_patches_<tile_size>/<slide_id>/`,
so the two scales never overwrite each other. The script skips a slide
if its `save_patches_<tile_size>/<slide_id>/` already exists, so re-runs
after a crash are cheap.

The screening rule (paper):
`mean < 236 ∧ std > 10 ∧ center_mean < 236`. The center crop is half the
tile on each side (`tile_size // 2` square), so for 256-px tiles it's
128×128 and for 128-px tiles it's 64×64 — same ratio across scales.
Override the thresholds with `--mean_thr / --std_thr / --center_mean_thr`
if you transfer to a different cohort.

DeepZoom level: pass `--dz_level_offset` (default `-3`, i.e.
third-highest level) to control which DeepZoom level is sampled. The
same level is now used both for `level_tiles[]` (cols/rows) and
`get_tile()` — the original hard-coded `level=15` only matched cohorts
whose slides had exactly 18 DeepZoom levels.

### Known caveats in `sliding_window.py`

- The `discard_patches_<tile_size>/` directory is created but never
  written to; it's benign but dead — kept for compatibility with the
  original layout.

## Step P2 — Subsample + class layout + 5-fold split

`subsample_and_layout.py` consumes the per-slide tile dumps from P1
plus a slide-level label CSV, and emits the directory tree that
`0-feature_extraction/run.py` reads.

Required label CSV (`labels.csv`):

```csv
slide_id,label
WSI-001,cancer
WSI-002,benign
WSI-003,atypical
WSI-004,suspicious
...
```

Allowed labels (case-insensitive on input, lowercased on disk):
`cancer`, `benign`, `atypical`, `suspicious`. The Step 0 dataset
wrapper matches these names exactly.

Run once for both scales:

```bash
python subsample_and_layout.py \
  --patches-roots /path/to/cohort/save_patches_256 /path/to/cohort/save_patches_128 \
  --out-roots    /path/to/cohort/scale256_5fold /path/to/cohort/scale128_5fold \
  --label-csv    /path/to/cohort/labels.csv \
  --n-patches 100 \
  --n-folds 5 \
  --seed 0 \
  --link-mode symlink
```

What it does:

- **Subsample**: for each slide, pick 100 patches without replacement;
  seed is `f"{--seed}:{slide_id}"` so it's deterministic and
  reproducible across reruns. Slides with fewer than 100 informative
  patches are dropped by default (`--on-too-few skip`); pass
  `--on-too-few oversample` if you instead want to sample with
  replacement.
- **Class layout**: the CSV maps `slide_id → class`; the script fails
  loudly if any slide on disk is missing from the CSV or has an unknown
  label.
- **5-fold split**: `StratifiedKFold(n_splits=5, shuffle=True,
  random_state=--seed)` over the surviving slides. The split is
  computed **once** and applied to every `--out-roots`, so the 128 /
  256 trees see the same fold assignments — this is what Step 1's
  cross-attention expects.
- **Materialisation**: by default uses **symlinks** to avoid a 5× disk
  blowup (each slide appears in 1 test fold + 4 train folds). Pass
  `--link-mode copy` if your downstream tooling doesn't follow
  symlinks.

The two `--out-roots` are then exactly what you pass to `run.py`:

```bash
# scale 256
python run.py --dataset urine --scale 256 \
  --slide_root /path/to/cohort/scale256_5fold \
  --nth_fold 0 --epochs 30 --VPUep 10 ...

# scale 128
python run.py --dataset urine --scale 128 \
  --slide_root /path/to/cohort/scale128_5fold \
  --nth_fold 0 --epochs 30 --VPUep 10 ...
```

## Sanity checks

After P2 finishes, before launching Step 0:

```bash
# Each fold's train should have ~ (n_train_slides * 100) patches.
find <out_root>/0/train -name '*.png' | wc -l

# Every slide should appear in test exactly once across the 5 folds.
python - <<'PY'
import json, collections
sp = json.load(open('<out_root>/splits.json'))
c = collections.Counter()
for f, d in sp.items():
    for s in d['test']:
        c[s] += 1
bad = [s for s, n in c.items() if n != 1]
assert not bad, f'slides with wrong test-count: {bad[:5]}'
print('ok')
PY

# 128 and 256 trees must share the exact same slide set.
diff <(find <scale256_out>/0 -mindepth 4 -maxdepth 4 -type d -printf '%f\n' | sort -u) \
     <(find <scale128_out>/0 -mindepth 4 -maxdepth 4 -type d -printf '%f\n' | sort -u)
```
