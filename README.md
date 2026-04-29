# LESS-WSI
The official implementation of paper "LESS: Label-efficient multi-scale learning for cytological whole slide image screening" accepted at Medical Image Analysis

## Abstract
In computational pathology, multiple instance learning (MIL) is widely used to circumvent the computational impasse in giga-pixel whole slide image (WSI) analysis. It usually consists of two stages: patch-level feature extraction and slide-level aggregation. Recently, pretrained models or self-supervised learning have been used to extract patch features, but they suffer from low effectiveness or inefficiency due to overlooking the task-specific supervision provided by slide labels. Here we propose a weakly-supervised Label-Efficient WSI Screening method, dubbed LESS, for cytological WSI analysis with only slide-level labels, which can be effectively applied to small datasets. First, we suggest using variational positive-unlabeled (VPU) learning to uncover hidden labels of both benign and malignant patches. We provide appropriate supervision by using slide-level labels to improve the learning of patch-level features. Next, we take into account the sparse and random arrangement of cells in cytological WSIs. To address this, we propose a strategy to crop patches at multiple scales and utilize a cross-attention vision transformer (CrossViT) to combine information from different scales for WSI classification. The combination of our two steps achieves task-alignment, improving effectiveness and efficiency. We validate the proposed label-efficient method on a urine cytology WSI dataset encompassing 130 samples (13,000 patches) and a breast cytology dataset FNAC 2019 with 212 samples (21,200 patches). The experiment shows that the proposed LESS reaches 84.79%, 85.43%, 91.79% and 78.30% on the urine cytology WSI dataset, and 96.88%, 96.86%, 98.95%, 97.06% on the breast cytology high-resolution-image dataset in terms of accuracy, AUC, sensitivity and specificity. It outperforms state-of-the-art MIL methods on pathology WSIs and realizes automatic cytological WSI cancer screening.

## Usage
The following commands are examples of running the code for in-house urine cytology dataset (will update the FNAC dataset soon).

## Preprocessing

> **Note.** The exact parameters in our preprocessing (DeepZoom level,
> intensity thresholds, the specific sliding-window screening rule)
> were tuned on our private urine cytology cohort. They are **not**
> required by the rest of the LESS pipeline. The only invariants the
> method actually needs are:
>
> - **100 informative patches per slide** (random sampling is fine), and
> - a **256×256 patch** plus the **central 128×128 crop** of that same
>   patch as the two scales fed into Step 0.
>
> On a new dataset you have three reasonable options for getting there:
> (a) reuse [CLAM's](https://github.com/mahmoodlab/CLAM) WSI segmentation
> + patching with whatever foreground threshold suits your stain, then
> randomly subsample to 100 / slide; (b) design your own screening
> heuristic; or (c) reuse our scripts as a starting point and just
> retune their thresholds for your scanner and stain.

Whichever route you take, the output that Step 0 expects is the
following per-scale 5-fold tree:

```
<slide_root>/
├── 0/                            # fold 0 (folds 1..4 have the same structure)
│   ├── train/
│   │   ├── cancer/<slide_id>/<patch>.png
│   │   ├── benign/<slide_id>/<patch>.png
│   │   ├── atypical/<slide_id>/<patch>.png
│   │   └── suspicious/<slide_id>/<patch>.png
│   └── test/
│       └── ... (same four classes)
└── splits.json                   # the realised slide-level split (same across scales)
```

Produce one such tree per scale (e.g. `scale256_5fold/`,
`scale128_5fold/`) and pass each as `--slide_root` in Step 0.

If you want the exact recipe we used (sliding-window thresholds,
DeepZoom level, the 100-patch subsample + stratified 5-fold layout
script), see
[`0-feature_extraction/PREPROCESSING.md`](0-feature_extraction/PREPROCESSING.md).

## 0 - Patch Feature Extraction

### What this step does

Step 0 trains a **Variational Positive-Unlabeled (VPU)** patch classifier
from slide-level labels only and then dumps its penultimate-layer features
for every patch. Because the cells are sparsely / randomly distributed in
cytology WSIs, we crop patches at **two scales** (e.g. 128 px and 256 px)
and run this step **twice** — once per scale. The two resulting feature
banks feed into the cross-attention aggregator in Step 1.

### Data folder structure

`--slide_root` should point to a per-scale patch dump that has **already
been split into the 5 CV folds** (the dataset wrapper indexes folds by
directory name, not by a split file):

```
<slide_root>/                        # e.g. PATH_TO_SAVED_SCALE128_PATCHES
└── 0/                               # fold 0 (folds 1..4 have the same structure)
    ├── train/
    │   ├── cancer/<slide_id>/<patch>.png
    │   ├── benign/<slide_id>/<patch>.png
    │   ├── atypical/<slide_id>/<patch>.png
    │   └── suspicious/<slide_id>/<patch>.png
    └── test/
        ├── cancer/<slide_id>/...
        └── ...
```

The split is read as `os.path.join(slide_root, str(nth_fold), 'train' | 'test')`
in `0-feature_extraction/dataset/dataset_urine.py`. Each slide has ~100
patches (paper).

**VPU training only uses `cancer` and `benign` slides.** The dataset
wrapper hard-codes `valid_classes=['benign', 'cancer']` for the labeled /
unlabeled VPU loaders, so any `atypical` or `suspicious` patches present
under a fold's `train/` are ignored during VPU training. They are still
embedded at feature-extraction time, so `<class>/<slide_id>` directories
for all four classes do appear under `--feature_root` afterwards (Step 1's
aggregator binarizes `cancer ∪ suspicious` vs. `benign ∪ atypical`).

In VPU terms, **`benign` is the labeled positive class** (`p_loader`) and
**`cancer` is the unlabeled class** (`x_loader`). Concretely, ImageFolder
sorts class names alphabetically (`benign=0`, `cancer=1`) and
`run.py`'s `--positive_label_list = [0]` selects `benign` as the positive
side. Slide-level labels are inferred from the parent class folder.

Note: `run.py`'s default `--slide_root` points to an old absolute path
(`/bigdata/projects/beidi/...`) — **always override it**.

### Train VPU + extract features (one scale at a time)

Run from `0-feature_extraction/`. Train for `--epochs N`. Each epoch writes
`<save_dir>/<epoch>.pth` containing the best-validation-so-far state dict
(lowest `val_var` up to and including that epoch). With `--get_feature 1`
the script then explicitly reloads `<save_dir>/<VPUep>.pth` (i.e. the
best-val checkpoint as of epoch `--VPUep`) and writes per-patch features
to `--feature_root` — so `--VPUep` acts as an early-stop index.

```bash
cd 0-feature_extraction

# scale 128
python run.py \
  --dataset urine \
  --scale 128 \
  --slide_root PATH_TO_SAVED_SCALE128_PATCHES \
  --nth_fold 0 \
  --epochs 30 --VPUep 10 \
  --batch-size 100 --lr 1e-4 \
  --alpha 0.3 --lam 0.03 --th 0.5 \
  --num_labeled 3000 \
  --save_dir ./save \
  --feature_root ./saved_feature \
  --get_feature 1 \
  --seed 0 --gpu 0

# scale 256
python run.py \
  --dataset urine \
  --scale 256 \
  --slide_root PATH_TO_SAVED_SCALE256_PATCHES \
  --nth_fold 0 \
  --epochs 30 --VPUep 10 \
  --batch-size 100 --lr 1e-4 \
  --alpha 0.3 --lam 0.03 --th 0.5 \
  --num_labeled 3000 \
  --save_dir ./save \
  --feature_root ./saved_feature \
  --get_feature 1 \
  --seed 0 --gpu 0
```

> **Important:** `run.py`'s default `--epochs 0` means *do not train, only
> extract*. Set `--epochs >= --VPUep` (e.g. `30` and `10`) to actually train
> the VPU. Set `--get_feature 0` if you only want to train, `1` (default) to
> also dump features.

### Key arguments

Defined in `0-feature_extraction/run.py`:

| flag | default | meaning |
|---|---|---|
| `--dataset` | `urine` | one of `urine`, `FANC`. Selects the dataset wrapper + label list. |
| `--scale` | `256` | patch crop size; pass `128` and `256` in two separate runs. |
| `--slide_root` | (private path) | root of patches for the chosen scale. **Override.** |
| `--nth_fold` | `0` | 0-indexed CV fold (5-fold split). |
| `--seed` | `0` | RNG seed. |
| `--epochs` | `0` | total VPU training epochs. **Set >= `--VPUep`** (e.g. 30) — leaving the default skips training. |
| `--VPUep` | `10` | early-stop checkpoint index used for feature extraction. The training loop writes one `.pth` per epoch named `<epoch>.pth`, but each file actually contains the **best-validation-so-far** state dict (lowest `val_var` over all epochs ≤ that epoch). Setting `--VPUep N` therefore loads "the best VPU model seen during the first N+1 epochs". The paper uses `N=10`. |
| `--batch-size` | `100` | patches per batch (matches 100 patches/slide). |
| `--lr` | `1e-4` | learning rate. |
| `--num_labeled` | `3000` | number of labeled positive patches used by VPU. |
| `--alpha` | `0.3` | Mixup parameter (VPU augmentation). |
| `--lam` | `0.03` | weight of the VPU regulariser. |
| `--th` | `0.5` | decision threshold for VPU pseudo-labels. |
| `--get_feature` | `1` | `1` = after training, reload VPU and dump features; `0` = train only. |
| `--feature_batch_size` | `128` | Batch size used when feeding patches through the trained VPU for feature dumping. The full DataLoader is now consumed (no per-batch break), so any value works; 128 conveniently fits a 100-patch slide in a single batch. |
| `--get_label` | (flag) | toggle to also write VPU pseudo-labels per patch. |
| `--save_dir` | `./save` | directory for VPU checkpoints (`<save_dir>/<epoch>.pth`). |
| `--feature_root` | `./saved_feature` | directory for the per-patch feature banks consumed by Step 1. |
| `--val-iterations` | `30` | validation iterations per epoch. |
| `--gpu` | `0` | GPU id. |

### Outputs

- `--save_dir` (default `./save/`) — VPU checkpoints, named `<epoch>.pth`.
- `--feature_root` (default `./saved_feature/`) — per-slide / per-patch
  features used by Step 1; one bank per scale × fold × seed.
- Point Step 1's `--data-path` at `--feature_root` (with the matching scale
  and fold). The two scales' feature dims correspond to Step 1's
  `--vpu_dim`, e.g. `384_768` for `crossvit_base_224`.

### Hyperparameters used in the paper

Final settings reported in the paper for the urine cytology dataset (use these
as defaults; the example commands above already match):

| flag | paper value |
|---|---|
| `--lr` | `1e-4` |
| `--batch-size` | `100` |
| `--epochs` | `30` |
| `--VPUep` | `10` |
| `--alpha` (Mixup) | `0.3` |
| `--lam` (VPU regulariser) | `0.03` |
| `--num_labeled` | `3000` |
| `--th` (decision threshold) | `0.5` |

If you transfer to a new dataset, search around these values
(e.g. `--lr ∈ {1e-3, 1e-4, 1e-5}`, `--alpha ∈ {0.1, 0.3, 0.5}`,
`--lam ∈ {0.01, 0.03, 0.1}`, `--num_labeled ∈ {1000, 3000, 5000}`,
`--th ∈ {0.3, 0.5, 0.7}`) on fold 0 / seed 0.

## 1 - Cross-attention-based Aggregation

After Step 0 produces per-patch features at two scales (e.g. `128`, `256`)
saved under `0-feature_extraction/saved_feature/`, Step 1 trains a
**multi-scale cross-attention vision transformer (CrossViT)** that fuses the
two scales into a single slide-level prediction.

### What the model does

The aggregator (`1-WSI_aggregation/models/crossvit.py`) keeps two parallel
transformer branches — one per scale — each with its own CLS token, then
fuses them with a `CrossAttentionBlock` where one branch's CLS token attends
over the other branch's patch tokens (and vice versa). The two final CLS
logits are averaged for the slide-level binary classification head. See the
paper for the full architecture diagram:
[arXiv:2306.03407](https://arxiv.org/pdf/2306.03407).

### Train / evaluate

Run from `1-WSI_aggregation/`. The default model is `crossvit_base_224`
(`embed_dim=[384, 768]`, `depth=[[1,4,0]]×3`, `num_heads=[16,16]`,
`mlp_ratio=[4,4,1]`, `qkv_bias=True`). Patch features from Step 0 are loaded
through `--features VPU` (use `PLIP` only if you swap in 512-dim PLIP
features — the model has `fc1/fc2` linear projectors for that case).

Training:

```bash
cd 1-WSI_aggregation
python main.py \
  --model crossvit_base_224 \
  --features VPU \
  --vpu_dim 384_768 \
  --data_set urine \
  --data-path PATH_TO_SAVED_FEATURES \
  --nth_fold 0 \
  --batch-size 1 \
  --epochs 100 \
  --opt adam \
  --lr 1e-6 \
  --warmup-lr 1e-6 \
  --warmup-epochs 1 \
  --sched cosine \
  --weight-decay 0.1 \
  --drop 0.1 \
  --seed 0 \
  --device cuda:0 \
  --output_dir ./outputs/crossvit_base_fold0
```

Evaluation only (load a checkpoint):

```bash
python main.py \
  --model crossvit_base_224 \
  --features VPU \
  --vpu_dim 384_768 \
  --data_set urine \
  --data-path PATH_TO_SAVED_FEATURES \
  --nth_fold 0 \
  --resume ./outputs/crossvit_base_fold0/checkpoint.pth \
  --eval
```

### Key arguments

Defined in `1-WSI_aggregation/main.py` (`get_args_parser`):

| flag | default | meaning |
|---|---|---|
| `--model` | `crossvit_base_224` | one of `crossvit_{tiny,small,base,large,9,15,18}_224` (`_dagger` variants use `multi_conv` patch embed). See `models/crossvit.py` for the full list. |
| `--features` | `VPU` | `VPU` for the VPU features from Step 0; `PLIP` activates the `fc1/fc2` 512→embed_dim projectors for PLIP features. |
| `--vpu_dim` | `384_768` | underscore-separated embedding dims of the two scales; must match the model's `embed_dim`. |
| `--input-size` | `240` | input size of the larger branch (the smaller branch is fixed at 224 in the model defs). |
| `--batch-size` | `1` | one slide per step (each slide is a bag of 100 patches). |
| `--epochs` | `1` (use ≥100) | total training epochs. |
| `--opt` | `adam` | optimizer (created via `timm.optim.create_optimizer`). |
| `--lr` | `1e-6` | base learning rate (paper uses `1e-6`–`1e-4`). |
| `--warmup-lr` / `--warmup-epochs` | `1e-6` / `1` | warmup schedule. |
| `--sched` | `cosine` | LR schedule (`timm.scheduler.create_scheduler`). |
| `--weight-decay` | `0.1` | AdamW-style weight decay. |
| `--drop` / `--drop-path` | `0.1` / `0` | dropout and stochastic-depth rate inside the transformer. |
| `--smoothing` | `0` | label smoothing for cross-entropy. |
| `--data_set` | `urine` | one of `urine`, `CIFAR10`, `CIFAR100`, `IMNET`, `INAT`, `INAT19`. |
| `--data-path` | — | root of saved Step-0 features. |
| `--nth_fold` | `0` | 0-indexed CV fold (5-fold split is built in `datasets.py`). |
| `--seed` | `42` | RNG seed. |
| `--initial_checkpoint` | (path) | optional path to a CrossViT ImageNet checkpoint (`crossvit_*_224.pth`) loaded with `--pretrained`. |
| `--resume` | `''` | resume / load checkpoint for `--eval`. |
| `--output_dir` | `''` | where to dump checkpoints + logs (must be set to keep checkpoints). |
| `--device` | `cuda:0` | training device. |

The full list (mixup, cutmix, color-jitter, repeated-aug, etc.) follows the
DeiT/timm conventions and is rarely needed for cytology features.

### Hyperparameters used in the paper

Final settings reported in the paper for CrossViT aggregation on the urine
cytology dataset (the training command above uses exactly these):

| flag | paper value |
|---|---|
| `--model` | `crossvit_base_224` |
| `--vpu_dim` | `384_768` |
| `--features` | `VPU` |
| `--batch-size` | `1` (one slide / bag per step) |
| `--epochs` | `100` |
| `--opt` | `adam` |
| `--lr` | `1e-6` |
| `--warmup-lr` | `1e-6` |
| `--warmup-epochs` | `1` |
| `--sched` | `cosine` |
| `--weight-decay` | `0.1` |
| `--drop` | `0.1` |

For a fresh dataset, search around these defaults
(e.g. `--lr ∈ {1e-4, 1e-5, 1e-6}`, `--weight-decay ∈ {0, 0.05, 0.1}`,
`--drop ∈ {0.0, 0.1, 0.3}`, `--model ∈ {crossvit_small_224, crossvit_base_224}`).

### Outputs

Training writes to `--output_dir`:

- `checkpoint.pth` (best on validation AUC)
- `log.txt` (per-epoch metrics)
- `wandb/` if W&B is enabled in `main.py`

Evaluation prints accuracy / AUC / sensitivity / specificity to stdout.


## Citation
If you find LESS-WSI useful for your research and applications, please cite using this BibTeX:
```bash
@article{zhao2024less,
  title={LESS: Label-efficient multi-scale learning for cytological whole slide image screening},
  author={Zhao, Beidi and Deng, Wenlong and Li, Zi Han Henry and Zhou, Chen and Gao, Zuhua and Wang, Gang and Li, Xiaoxiao},
  journal={Medical Image Analysis},
  volume={94},
  pages={103109},
  year={2024},
  publisher={Elsevier}
}
```
