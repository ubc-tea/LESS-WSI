# Preprocessing spec: from `sliding_window.py` to Step 0 input

This document describes everything that has to happen between the current
output of `sliding_window.py` and the on-disk layout that Step 0
(`run.py` / `dataset/dataset_urine.py`) expects to consume. Treat it as a
checklist for writing the missing glue scripts — `sliding_window.py` itself
is left untouched.

## 0. What the paper / Step 0 actually need

Per the paper:

- Two scales of patches per slide: **128 px** and **256 px**
- Each WSI contributes **100 randomly sampled informative patches** per scale
- 5-fold cross-validation across slides
- Each patch's slide-level class label is one of:
  `cancer`, `benign`, `atypical`, `suspicious`
- For VPU, only `cancer` is positive (`--positive_label_list = [1]`); the
  other three are unlabeled

Step 0 reads from disk via:

```python
os.path.join(slide_root, str(nth_fold), 'train' | 'test',
             '<class>', '<slide_id>', '<patch>.png')
```

So the final on-disk shape (per scale) must be:

```
<slide_root_scale_S>/                 # one root per scale (S ∈ {128, 256})
└── <fold>/                            # fold ∈ {0,1,2,3,4}
    ├── train/
    │   ├── cancer/<slide_id>/<patch>.png
    │   ├── benign/<slide_id>/<patch>.png
    │   ├── atypical/<slide_id>/<patch>.png
    │   └── suspicious/<slide_id>/<patch>.png
    └── test/
        └── ... (same four classes)
```

`<slide_root_scale_S>` is what you pass as `--slide_root` to `run.py`, with
`--scale S` set to match.

## 1. What `sliding_window.py` produces today

```
<wsi_path>/
├── images/<slide_id>.svs            # input
└── save_patches/
    └── <slide_id>/
        └── <col>_<row>_m.png         # all informative 256-px tiles
```

Properties of the existing output:

- **Scale**: 256 px only (`tile_size=256` is hard-coded).
- **Filter**: `mean<236 ∧ std>10 ∧ center_mean<236` — already applied,
  the kept tiles are the "informative" set.
- **Count per slide**: variable; can be a few hundred. Not the 100 the
  paper uses.
- **Class label**: not encoded. Tiles are organised only by `slide_id`.
- **Fold assignment**: not encoded.
- **DeepZoom level note**: `level_tiles[-3]` is used to compute `cols/rows`
  but `get_tile(15, ...)` is written with a literal level `15`. This is a
  potential bug worth verifying on your slides (whether `15 == len(levels)-3`
  for every WSI in the cohort). Not a "must fix to align with Step 0", but
  a sanity-check you should run before generating the final dataset.

## 2. Gap list

To bridge §1 → §0, the following four transformations are required. They
can be implemented as three separate scripts run in sequence, or as one
end-to-end pipeline.

### Gap A: produce 128-px patches as well

Current pipeline only emits 256-px tiles. Two viable strategies:

- **A1 (preferred, matches paper intent)**: re-run a 128-px sliding window
  (`tile_size=128`, same overlap=0) over each `.svs` with the same
  `mean/std/center_mean` filter, writing to a parallel
  `<wsi_path>/save_patches_128/<slide_id>/`. Keeps both scales independent.
- **A2 (cheap approximation)**: take the existing 256-px tiles and crop /
  resize them to 128. This is **not** what the paper does (the two scales
  are meant to look at different cell neighbourhoods), so only use this if
  you cannot re-run sliding window.

After this step you should have two parallel patch dumps:

```
<wsi_path>/save_patches/      # 256 px (already exists)
<wsi_path>/save_patches_128/  # 128 px (new)
```

The rest of the spec applies to **both** dumps independently.

### Gap B: random subsample to 100 patches per slide

For every `<slide_id>` directory under `save_patches[_128]/`:

- If `len(tiles) >= 100`: sample 100 without replacement.
- If `len(tiles) <  100`: either skip the slide (paper-faithful, drops the
  slide from the cohort) or oversample with replacement (not paper-faithful;
  flag it in logs). Decide once and document.

Sampling RNG should be **seeded** (e.g. seed = `hash(slide_id)`) so the
subsample is reproducible and identical across the 128 / 256 dumps if you
want spatially-aligned subsets. Note: even with the same seed you cannot
get the same *spatial* tiles between scales because the tile grids differ;
"alignment" here only means same RNG state, not same content.

Output (in place or to a new directory):

```
<wsi_path>/save_patches_100/<slide_id>/<patch>.png         # 256-px subset
<wsi_path>/save_patches_128_100/<slide_id>/<patch>.png     # 128-px subset
```

### Gap C: attach the slide-level class label

`sliding_window.py` does not know the class. You need an **external label
file** mapping every `slide_id` to one of `{cancer, benign, atypical,
suspicious}`. Typical formats:

```csv
slide_id,label
WSI-001,cancer
WSI-002,benign
...
```

The script that consumes this CSV moves/copies each
`save_patches_100/<slide_id>/` into the matching class subdirectory. After
this step:

```
<wsi_path>/by_class_256/
├── cancer/<slide_id>/<patch>.png
├── benign/<slide_id>/<patch>.png
├── atypical/<slide_id>/<patch>.png
└── suspicious/<slide_id>/<patch>.png
```

(and the same under `by_class_128/`).

Validation: count unique slide_ids in the CSV vs. unique slide_ids on disk;
warn on any mismatch.

### Gap D: 5-fold split → final Step-0 layout

Generate a 5-fold split **at slide level**, not patch level (otherwise
patches from the same slide would leak across train/test).

- **Stratify** by class (`cancer/benign/atypical/suspicious`) so every
  fold has roughly the same class ratio. Use
  `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True,
  random_state=0)` or equivalent.
- The split must be **identical across the 128 and 256 dumps** so that a
  given fold's train/test slides match between the two scales (Step 1's
  cross-attention assumes the same slide is represented in both).

For each fold f ∈ {0,1,2,3,4}, write:

```
<slide_root_scale_S>/<f>/train/<class>/<slide_id>/<patch>.png
<slide_root_scale_S>/<f>/test/<class>/<slide_id>/<patch>.png
```

You may want to use **symlinks** rather than copies to avoid a 5× disk
blowup (each slide ends up in 1 test-fold and 4 train-folds).

Persist the split itself (e.g. `splits.json` containing
`{fold: {train: [...slide_ids], test: [...slide_ids]}}`) alongside the
data so it can be reproduced and audited.

## 3. Suggested script breakdown

A reasonable refactor is three scripts in `0-feature_extraction/`:

| script | role | inputs | outputs |
|---|---|---|---|
| `sliding_window.py` (existing) | informative-tile filter at one scale | `--wsi_path`, hard-coded `tile_size=256` | `save_patches/<slide_id>/*.png` |
| `sliding_window.py` re-run with `tile_size=128` (Gap A1) | same, 128-px scale | `--wsi_path`, `tile_size=128` | `save_patches_128/<slide_id>/*.png` |
| `subsample_and_layout.py` (new) | Gaps B + C + D in one pass per scale | label CSV, both `save_patches[_128]/` dirs, `--n_patches 100`, `--n_folds 5`, `--seed` | `<slide_root_scale_S>/<fold>/{train,test}/<class>/<slide_id>/<patch>.png` for S ∈ {128, 256} |

The subsample/layout script is the only **new** piece of code required; the
two `sliding_window.py` invocations are existing-code re-runs.

## 4. Wiring back into Step 0

Once Gaps A–D are done, the Step 0 commands in the README work directly:

```bash
# 256 scale
python run.py --dataset urine --scale 256 \
  --slide_root <slide_root_scale_256> \
  --nth_fold 0 --epochs 30 --VPUep 10 ...

# 128 scale
python run.py --dataset urine --scale 128 \
  --slide_root <slide_root_scale_128> \
  --nth_fold 0 --epochs 30 --VPUep 10 ...
```

`--positive_label_list` defaults to `[1]` (i.e. `cancer`), matching the
class folders produced in Gap C.

## 5. Sanity checks before launching Step 0

- `find <slide_root_scale_S>/0/train -name '*.png' | wc -l` ≈
  `n_train_slides * 100` per fold.
- Every `<slide_id>` should appear in exactly **one** of `{train, test}`
  per fold and in **all 5 folds** combined as a test slide once.
- `set(slides_in_128_dump) == set(slides_in_256_dump)` for every fold and
  split (otherwise the cross-attention can't pair the two scales).
- Class folder names exactly: `cancer`, `benign`, `atypical`,
  `suspicious` (case-sensitive in `dataset_urine.py`).

## 6. Things `sliding_window.py` already does correctly (do not redo)

- The 256-px patch size and `overlap=0`.
- The white-background filter (`mean<236 ∧ std>10 ∧ center_mean<236`).
- Per-slide directory layout (`save_patches/<slide_id>/`).
- Skip-on-exists guard (`if os.path.exists(save_dir): continue`) — useful
  for re-runs after partial failures.

These should be preserved when re-running the script for the 128-px scale.
