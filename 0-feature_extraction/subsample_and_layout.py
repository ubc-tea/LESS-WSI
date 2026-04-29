"""
Subsample and re-layout per-slide patches into the directory structure that
Step 0 (`run.py` / `dataset/dataset_urine.py`) consumes.

Implements Gaps B, C, D from PREPROCESSING_SPEC.md:
  B. Random subsample to N (default 100) patches per slide (seeded).
  C. Attach slide-level class label via an external CSV.
  D. Stratified 5-fold split at slide level, written as
     <out_root>/<fold>/{train,test}/<class>/<slide_id>/<patch>.png

The same split is applied to multiple scales (e.g. 128 and 256) so that the
per-fold train/test slide sets are identical across scales, which is what
the Step 1 cross-attention assumes.

Inputs:
  --patches-roots   one or more roots, each with <slide_id>/<patch>.png inside
                    (typically: save_patches/ for 256, save_patches_128/ for 128)
  --out-roots       parallel list of destination roots (one per --patches-roots).
                    Each becomes a valid `--slide_root` for run.py at that scale.
  --label-csv       CSV with columns slide_id,label
  --n-patches       per-slide random subsample size (default 100)
  --n-folds         stratified K-fold count (default 5)
  --seed            RNG seed for both subsampling and the K-fold split
  --link-mode       'symlink' (default, recommended) or 'copy'
  --on-too-few      what to do if a slide has < n-patches: 'skip' (default,
                    paper-faithful) or 'oversample'

Outputs:
  <out_root>/<fold>/{train,test}/<class>/<slide_id>/<patch>.png   for each scale
  <out_root>/splits.json   the realised slide-level split (same content for
                           every scale because the seed is shared)

Run from 0-feature_extraction/. This script does not touch sliding_window.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

try:
    from sklearn.model_selection import StratifiedKFold
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for the stratified split. "
        "Install with `pip install scikit-learn`."
    ) from e


VALID_CLASSES = ("cancer", "benign", "atypical", "suspicious")


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def load_label_csv(path: Path) -> dict[str, str]:
    """Read slide_id,label CSV. Reject unknown labels."""
    labels: dict[str, str] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "slide_id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"{path} must have columns 'slide_id' and 'label'; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            sid = row["slide_id"].strip()
            lab = row["label"].strip().lower()
            if not sid:
                continue
            if lab not in VALID_CLASSES:
                raise ValueError(
                    f"{path}: slide {sid} has label '{lab}', "
                    f"expected one of {VALID_CLASSES}"
                )
            if sid in labels and labels[sid] != lab:
                raise ValueError(
                    f"{path}: slide {sid} appears twice with conflicting labels "
                    f"({labels[sid]} vs {lab})"
                )
            labels[sid] = lab
    if not labels:
        raise ValueError(f"{path} contained no rows")
    return labels


def list_patches(slide_dir: Path) -> list[Path]:
    """All .png patches directly inside slide_dir, sorted for determinism."""
    return sorted(p for p in slide_dir.iterdir() if p.suffix.lower() == ".png")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """Create dst (parents made on demand) by symlinking or copying from src."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        # Re-running this script should be idempotent.
        dst.unlink()
    if mode == "symlink":
        # Use absolute target so the link survives moves of the out tree.
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown link-mode {mode!r}")


# --------------------------------------------------------------------------- #
# Core stages
# --------------------------------------------------------------------------- #

def subsample_slide(
    slide_id: str,
    patches: list[Path],
    n_patches: int,
    on_too_few: str,
    base_seed: int,
) -> list[Path] | None:
    """Return a length-n_patches deterministic subset, or None if skipped."""
    rng = random.Random(f"{base_seed}:{slide_id}")
    if len(patches) >= n_patches:
        return rng.sample(patches, n_patches)
    if on_too_few == "skip":
        return None
    if on_too_few == "oversample":
        # Sample with replacement up to n_patches.
        return [rng.choice(patches) for _ in range(n_patches)]
    raise ValueError(f"unknown on-too-few {on_too_few!r}")


def stratified_kfold(
    slide_ids: list[str],
    labels: dict[str, str],
    n_folds: int,
    seed: int,
) -> dict[int, dict[str, list[str]]]:
    """{fold: {'train': [...slide_ids], 'test': [...slide_ids]}}."""
    sids = sorted(slide_ids)  # deterministic order before shuffling
    y = [labels[s] for s in sids]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: dict[int, dict[str, list[str]]] = {}
    for f, (train_idx, test_idx) in enumerate(skf.split(sids, y)):
        splits[f] = {
            "train": [sids[i] for i in train_idx],
            "test": [sids[i] for i in test_idx],
        }
    return splits


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run(
    patches_roots: list[Path],
    out_roots: list[Path],
    label_csv: Path,
    n_patches: int,
    n_folds: int,
    seed: int,
    link_mode: str,
    on_too_few: str,
) -> None:
    if len(patches_roots) != len(out_roots):
        raise ValueError(
            f"--patches-roots and --out-roots must have the same length "
            f"(got {len(patches_roots)} vs {len(out_roots)})"
        )

    labels = load_label_csv(label_csv)

    # Pick the slide universe from the FIRST patches root, then verify that
    # every other root contains the same slides. This guarantees the 128 and
    # 256 dumps share a slide set (Step 1's cross-attention requires this).
    first_root = patches_roots[0]
    if not first_root.is_dir():
        raise FileNotFoundError(first_root)
    slide_dirs_first = sorted(p for p in first_root.iterdir() if p.is_dir())
    slide_ids_first = [p.name for p in slide_dirs_first]

    missing_label = [s for s in slide_ids_first if s not in labels]
    if missing_label:
        raise ValueError(
            f"{len(missing_label)} slide(s) in {first_root} not found in "
            f"label CSV (first 5: {missing_label[:5]})"
        )

    for root in patches_roots[1:]:
        if not root.is_dir():
            raise FileNotFoundError(root)
        sids = {p.name for p in root.iterdir() if p.is_dir()}
        diff_a = set(slide_ids_first) - sids
        diff_b = sids - set(slide_ids_first)
        if diff_a or diff_b:
            raise ValueError(
                f"slide sets differ between {first_root} and {root}: "
                f"missing in second={sorted(diff_a)[:5]}, "
                f"extra in second={sorted(diff_b)[:5]}"
            )

    # Stage B: subsample per slide, per scale, with the same RNG seed across
    # scales (so the script is reproducible; the actual file sets differ
    # across scales because the available patches differ).
    print(f"[stage B] subsampling up to {n_patches} patches per slide "
          f"({on_too_few=}, {seed=})", file=sys.stderr)
    per_scale_subsets: list[dict[str, list[Path]]] = []
    skipped_per_scale: list[list[str]] = []
    for root in patches_roots:
        subsets: dict[str, list[Path]] = {}
        skipped: list[str] = []
        for sid in slide_ids_first:
            patches = list_patches(root / sid)
            chosen = subsample_slide(sid, patches, n_patches, on_too_few, seed)
            if chosen is None:
                skipped.append(sid)
            else:
                subsets[sid] = chosen
        per_scale_subsets.append(subsets)
        skipped_per_scale.append(skipped)
        print(f"  {root}: kept {len(subsets)} slides, skipped {len(skipped)}",
              file=sys.stderr)

    # Drop any slide that fell below n_patches in ANY scale, so the same
    # cohort is used on both sides.
    keep_slides = sorted(
        set(slide_ids_first)
        .difference(*[set(s) for s in skipped_per_scale])
    )
    if len(keep_slides) < len(slide_ids_first):
        dropped = sorted(set(slide_ids_first) - set(keep_slides))
        print(f"[stage B] {len(dropped)} slide(s) dropped because they fell "
              f"below {n_patches} patches in at least one scale "
              f"(first 5: {dropped[:5]})", file=sys.stderr)
    if len(keep_slides) < n_folds:
        raise RuntimeError(
            f"only {len(keep_slides)} slide(s) survive subsampling; "
            f"cannot run {n_folds}-fold CV"
        )

    # Stage D: stratified K-fold at the slide level (shared across scales).
    print(f"[stage D] stratified {n_folds}-fold split over "
          f"{len(keep_slides)} slides", file=sys.stderr)
    splits = stratified_kfold(keep_slides, labels, n_folds, seed)
    for f, sp in splits.items():
        cls_counts = {c: 0 for c in VALID_CLASSES}
        for sid in sp["test"]:
            cls_counts[labels[sid]] += 1
        print(f"  fold {f}: train={len(sp['train'])} test={len(sp['test'])} "
              f"test_class_counts={cls_counts}", file=sys.stderr)

    # Stage C+D output: write each scale's tree, plus splits.json (identical
    # across scales — written once per out_root for self-containedness).
    for root, out_root, subsets in zip(patches_roots, out_roots, per_scale_subsets):
        print(f"[stage C+D] materialising {out_root} from {root} "
              f"(mode={link_mode})", file=sys.stderr)
        out_root.mkdir(parents=True, exist_ok=True)
        with (out_root / "splits.json").open("w") as f:
            json.dump(splits, f, indent=2)
        for fold, sp in splits.items():
            for split_name in ("train", "test"):
                for sid in sp[split_name]:
                    if sid not in subsets:
                        # Was dropped at stage B in this or another scale.
                        continue
                    cls = labels[sid]
                    if cls not in VALID_CLASSES:
                        raise RuntimeError(f"invariant: bad class {cls}")
                    dst_dir = out_root / str(fold) / split_name / cls / sid
                    for src in subsets[sid]:
                        dst = dst_dir / src.name
                        link_or_copy(src, dst, link_mode)
        print(f"  done {out_root}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Subsample + relayout patches into Step 0's expected tree."
    )
    p.add_argument(
        "--patches-roots", nargs="+", required=True, type=Path,
        help="One or more per-scale patch roots, each containing "
             "<slide_id>/<patch>.png. Order must match --out-roots.",
    )
    p.add_argument(
        "--out-roots", nargs="+", required=True, type=Path,
        help="One destination root per --patches-roots. Each becomes a valid "
             "`--slide_root` for run.py at the matching scale.",
    )
    p.add_argument(
        "--label-csv", required=True, type=Path,
        help="CSV with columns slide_id,label "
             "(label ∈ {cancer,benign,atypical,suspicious}).",
    )
    p.add_argument("--n-patches", type=int, default=100,
                   help="patches per slide after subsampling (default 100).")
    p.add_argument("--n-folds", type=int, default=5,
                   help="number of CV folds (default 5).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for both subsampling and the split (default 0).")
    p.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink",
                   help="how to materialise per-fold trees (default symlink).")
    p.add_argument("--on-too-few", choices=("skip", "oversample"), default="skip",
                   help="behaviour when a slide has < n-patches available "
                        "(default skip = paper-faithful).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        patches_roots=args.patches_roots,
        out_roots=args.out_roots,
        label_csv=args.label_csv,
        n_patches=args.n_patches,
        n_folds=args.n_folds,
        seed=args.seed,
        link_mode=args.link_mode,
        on_too_few=args.on_too_few,
    )


if __name__ == "__main__":
    main()
