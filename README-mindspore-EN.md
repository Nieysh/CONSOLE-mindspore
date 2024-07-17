# Correctable Landmark Discovery via Large Models for Vision-Language Navigation

Mindspore implementation of the paper [Correctable Landmark Discovery via Large Models for Vision-Language Navigation](https://arxiv.org/abs/2405.18721).

Vision-Language Navigation (VLN) requires the agent to follow language instructions to reach a target position. A key factor for successful navigation is to align the landmarks implied in the instruction with diverse visual observations. However, previous VLN agents fail to perform accurate modality alignment especially in unexplored scenes, since they learn from limited navigation data and lack sufficient open-world alignment knowledge. In this work, we propose a new VLN paradigm, called COrrectable LaNdmark DiScOvery via Large ModEls (CONSOLE). In CONSOLE, we cast VLN as an open-world sequential landmark discovery problem, by introducing a novel correctable landmark discovery scheme based on two large models ChatGPT and CLIP. Specifically, we use ChatGPT to provide rich open-world landmark cooccurrence commonsense, and conduct CLIP-driven landmark discovery based on these commonsense priors. To mitigate the noise in the priors due to the lack of visual constraints, we introduce a learnable cooccurrence scoring module, which corrects the importance of each cooccurrence according to actual observations for accurate landmark discovery. We further design an  observation enhancement strategy for an elegant combination of our framework with different VLN agents, where we utilize the corrected landmark features to obtain enhanced observation features for action decision. Extensive experimental results on multiple popular VLN benchmarks (R2R, REVERIE, R4R, RxR) show the significant superiority of CONSOLE over strong baselines. Especially, our CONSOLE establishes the new state-of-the-art results on R2R and R4R in unseen scenarios.

![framework](motivation.png)

## Prerequisites

### Installation

1. Install mindspore (python=3.9,cuda=11.6)
```
conda create -n mindspore_console_py39 python=3.9 -y
conda activate mindspore_console_py39
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0rc2/MindSpore/unified/x86_64/mindspore-2.3.0rc2-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. Install other requirements for console
```
pip install -r requirements.txt

# install mindformers
git clone -b r1.1.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh

# install mindnlp
pip install mindnlp==0.3.2

# install timm
git clone https://github.com/rwightman/pytorch-image-models.git
cd pytorch-image-models
git checkout 9cc7dda6e5fcbbc7ac5ba5d2d44050d2a8e3e38d
```

3. Follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator) to install Matterport3D simulators.


### Data Preparation
1. Download features, connectivity files, and trained models following instructions [here](https://github.com/cshizhe/VLN-HAMT).
2. Download annotation data containing sequential landmarks and landmark cooccurrences from [here](https://drive.google.com/drive/folders/15ofyBXUHrqklos7e6Gptn5fXleVjS7Hg?usp=sharing).

## Navigation
### VLN Inference

```setup
cd finetune_src
bash scripts/run_r2r_mindspore.sh
```

## Citation
If you find this work useful, please consider citing:
```setup
@article{lin2024correctable,
  title={Correctable Landmark Discovery Via Large Models for Vision-Language Navigation},
  author={Lin, Bingqian and Nie, Yunshuang and Wei, Ziming and Zhu, Yi and Xu, Hang and Ma, Shikui and Liu, Jianzhuang and Liang, Xiaodan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

