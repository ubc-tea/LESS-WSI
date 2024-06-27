# LESS-WSI
The official implementation of paper "LESS: Label-efficient multi-scale learning for cytological whole slide image screening" accepted at Medical Image Analysis

## Abstract
In computational pathology, multiple instance learning (MIL) is widely used to circumvent the computational impasse in giga-pixel whole slide image (WSI) analysis. It usually consists of two stages: patch-level feature extraction and slide-level aggregation. Recently, pretrained models or self-supervised learning have been used to extract patch features, but they suffer from low effectiveness or inefficiency due to overlooking the task-specific supervision provided by slide labels. Here we propose a weakly-supervised Label-Efficient WSI Screening method, dubbed LESS, for cytological WSI analysis with only slide-level labels, which can be effectively applied to small datasets. First, we suggest using variational positive-unlabeled (VPU) learning to uncover hidden labels of both benign and malignant patches. We provide appropriate supervision by using slide-level labels to improve the learning of patch-level features. Next, we take into account the sparse and random arrangement of cells in cytological WSIs. To address this, we propose a strategy to crop patches at multiple scales and utilize a cross-attention vision transformer (CrossViT) to combine information from different scales for WSI classification. The combination of our two steps achieves task-alignment, improving effectiveness and efficiency. We validate the proposed label-efficient method on a urine cytology WSI dataset encompassing 130 samples (13,000 patches) and a breast cytology dataset FNAC 2019 with 212 samples (21,200 patches). The experiment shows that the proposed LESS reaches 84.79%, 85.43%, 91.79% and 78.30% on the urine cytology WSI dataset, and 96.88%, 96.86%, 98.95%, 97.06% on the breast cytology high-resolution-image dataset in terms of accuracy, AUC, sensitivity and specificity. It outperforms state-of-the-art MIL methods on pathology WSIs and realizes automatic cytological WSI cancer screening.

## Usage
The following commands are examples of running the code for in-house urine cytology dataset (will update the FNAC dataset soon).

## Preprocessing
Although the preprocessing of [CLAM](https://github.com/mahmoodlab/CLAM/tree/master#wsi-segmentation-and-patching) is widely used for histopathology WSIs, it is not suitable to be directly used for cytology WSIs, which will lose a lot of patches. Therefore, we buld our own prepricessing pipeline to select informative patches. Since the patches are randomly located in the WSIs, we randomly sample 100 patches from each WSI.
- [ ] Upload the preprocessing file.

## 0 - Patch Feature Extraction
- [ ] Add the data folder structre.

To train thr VPU model on two scales and extract features with VPU model:
```bash
python run.py --scale 128 --slide_root PATH_TO_SAVED_SCALE128_PATCHES --nth_fold 0
python run.py --scale 256 --slide_root PATH_TO_SAVED_SCALE128_PATCHES --nth_fold 0
```
The feature is saved in the folder ```bash saved_feature```

## 1 - Cross-attention-based Aggregation
- [ ] Check the code.


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