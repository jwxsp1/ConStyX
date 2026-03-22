# [MICCAI 2025] ConStyX

This repo is the PyTorch implementation of our paper:

**["ConStyX: Content Style Augmentation for Generalizable Medical Image Segmentation"](https://arxiv.org/abs/2506.10675)**

<!-- ![framework](docs/ConStyX.png) -->
<img src=fig/constyx.png width=75% />

**Con**tent **Sty**le Augmentation (**ConStyX**)

## Usage

🔥🔥 Code for generalizable medical image segmentation with ConStyX. 🔥🔥
### 1. Data Preparation
Download the datasets from this link: [OD/OC Segmentation](https://zenodo.org/record/8009107)<br />
Then, the dataset is arranged in the following format:
```
dataset/
|-- ORIGA
|   |-- train
|   |   |-- images
|   |   |-- mask
|   |-- test
|   |   |-- image
|   |   |-- mask
...
```
### 2. OD/OC Segmentation
We take the scenario using BinRushed (target domain) and other four datasets (source domains) as the example.
```
# Training
CUDA_VISIBLE_DEVICES=0 python main.py --mode train_DG --num_epochs 100 --Source_Dataset BinRushed
# Test
CUDA_VISIBLE_DEVICES=0 python test.py --mode multi_test --load_time TIME_OF_MODEL --Source_Dataset BinRushed
```
## Citation
If you find this project useful, please consider citing:
```
@inproceedings{chen2025constyx,
  title={Constyx: Content style augmentation for generalizable medical image segmentation},
  author={Chen, Xi and Shen, Zhiqiang and Cao, Peng and Yang, Jinzhu and Zaiane, Osmar R},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={100--110},
  year={2025},
  organization={Springer}
}
```
## Acknowledgements
Part of the code is revised from the Pytorch implementation of [TriD](https://github.com/Chen-Ziyang/TriD)

Thanks to the authors for providing the processed data.
