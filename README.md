# LESS-WSI
The official implementation of paper "LESS: Label-efficient multi-scale learning for cytological whole slide image screening" accepted at Medical Image Analysis

## Abstract

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