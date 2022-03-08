# ACVNet
This is the implementation of the paper: **Attention Concatenation Volume for Accurate and Efficient Stereo Matching**, CVPR 2022, Gangwei Xu, Junda Cheng, Peng Guo, Xin Yang
[\[Arxiv\]](https://arxiv.org/)

![image](https://github.com/gangweiX/ACVNet/blob/main/imgs/acv.png)

# How to use

## Environment
* python 3.6
* Pytorch >= 0.4.1

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

# Acknowledgements

Thanks to Xiaoyang Guo for opening source of his excellent work GwcNet. Our work is inspired by this work and part of codes are migrated from [GwcNet](https://github.com/xy-guo/GwcNet).
