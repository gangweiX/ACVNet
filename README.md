# ACVNet
This is the implementation of the paper: **Attention Concatenation Volume for Accurate and Efficient Stereo Matching**, CVPR 2022, Gangwei Xu, Junda Cheng, Peng Guo, Xin Yang
[\[Arxiv\]](https://arxiv.org/pdf/2203.02146.pdf)

## Introduction

An informative and concise cost volume representation is vital for stereo matching of high accuracy and efficiency. In this paper, we present a novel cost volume construction method which generates attention weights from correlation clues to suppress redundant information and enhance matching-related information in the concatenation volume. To generate reliable attention weights, we propose multi-level adaptive patch matching to improve the distinctiveness of the matching cost at different disparities even for textureless regions.

![image](https://github.com/gangweiX/ACVNet/blob/main/imgs/acv_network.png)

# How to use

## Environment
* python 3.8
* Pytorch 1.10

## Install

### Create a virtual environment and activate it.

```
conda create -n acvnet python=3.8
conda activate acvnet
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install opencv-python
pip install tensorboard
pip install matplotlib 
pip install tqdm
```

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## Train
As an example, use the following command to train ACVNet on Scene Flow

```
python main.py
```

### Pretrained Model

The pretrained model on Scene Flow Datasets is saved in ./checkpoints/model_sceneflow.ckpt

## Results on KITTI 2015 leaderboard
[Leaderboard Link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-bg (All) | D1-fg (All) | D1-all (All) | Runtime (s) |
|---|---|---|---|---|
| ACVNet | 1.37 % | 3.07 % | 1.65 % | 0.20 |
| LEAStereo | 1.40 % | 2.91 % | 1.65 % | 0.30 |
| GwcNet | 1.74 % | 3.93 % | 2.11 % | 0.32 |
| PSMNet | 1.86 % | 4.62 % | 2.32 % | 0.41 |

## Qualitative results on Scene Flow Datasets, KITTI 2012 and KITTI 2015

### The left column is left image, and the right column is results of our ACVNet.

![image](https://github.com/gangweiX/ACVNet/blob/main/imgs/acv_result.png)

# Citarion

If you find this project helpful in your research, welcome to cite the paper.

```
@article{xu2022ACVNet,
  title={ACVNet: Attention Concatenation Volume for Accurate and Efficient Stereo Matching},
  author={Gangwei Xu, Junda Cheng, Peng Guo, Xin Yang},
  journal={arXiv:2203.02146},
  year={2022}
}

```

# Acknowledgements

Thanks to Xiaoyang Guo for opening source of his excellent work GwcNet. Our work is inspired by this work and part of codes are migrated from [GwcNet](https://github.com/xy-guo/GwcNet).
