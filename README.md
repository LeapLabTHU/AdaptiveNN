# AdaptiveNN (NMI'25)

[![DOI](https://zenodo.org/badge/1036530001.svg)](https://doi.org/10.5281/zenodo.16810995)
<a href='https://arxiv.org/abs/2509.15333'><img src='https://img.shields.io/badge/Arxiv-2509.15333-red'>



<p align="center">
<img src="assets/teaser.png" width=100% height=100% 
class="center">
</p>

<!-- This repo contains the official code and pre-trained models for the *Nature Machine Intelligence* paper **"[Emulating Human-like Adaptive Vision for Efficient and Flexible Machine Visual Perception](https://www.nature.com/articles/s42256-025-01130-7)"** -->

This repo contains the official code and pre-trained models for the paper **[AdaptiveNN](https://www.nature.com/articles/s42256-025-01130-7)**.

> **Title:** &emsp;&emsp;Emulating Human-like Adaptive Vision for Efficient and Flexible Machine Visual Perception
> 
> **Authors:**&nbsp; Yulin Wang(王语霖)<sup>†</sup>, Yang Yue(乐洋)<sup>†</sup>, Yang Yue(乐阳)<sup>†</sup>, Huanqian Wang, Haojun Jiang, Yizeng Han, Zanlin Ni, Yifan Pu, Minglei Shi, Rui Lu, Qisen Yang, Andrew Zhao, Zhuofan Xia, Shiji Song<sup>#</sup>, Gao Huang<sup>#</sup>. (<sup>†</sup> Equal Contribution, <sup>#</sup> Corresponding Author)
> 
> **Institute:** &nbsp;Department of Automation, Tsinghua University
> 
> **Publish:** &nbsp;&nbsp;Nature Machine Intelligence 2025
>

## Abstract
Human vision is highly adaptive, efficiently sampling intricate environments by sequentially fixating on task-relevant regions. In contrast, prevailing machine vision models passively process entire scenes at once, resulting in excessive resource demands scaling with spatial–temporal input resolution and model size, yielding critical limitations impeding both future advancements and real-world application. Here we introduce AdaptiveNN, a general framework aiming to enable the transition from ‘passive’ to ‘active and adaptive’ vision models. AdaptiveNN formulates visual perception as a coarse-to-fine sequential decision-making process, progressively identifying and attending to regions pertinent to the task, incrementally combining information across fixations and actively concluding observation when sufficient. We establish a theory integrating representation learning with self-rewarding reinforcement learning, enabling end-to-end training of the non-differentiable AdaptiveNN without additional supervision on fixation locations. We assess AdaptiveNN on 17 benchmarks spanning 9 tasks, including large-scale visual recognition, fine-grained discrimination, visual search, processing images from real driving and medical scenarios, language-driven embodied artificial intelligence and side-by-side comparisons with humans. AdaptiveNN achieves up to 28 times inference cost reduction without sacrificing accuracy, flexibly adapts to varying task demands and resource budgets without retraining, and provides enhanced interpretability via its fixation patterns, demonstrating a promising avenue towards efficient, flexible and interpretable computer vision. Furthermore, AdaptiveNN exhibits closely human-like perceptual behaviours in many cases, revealing its potential as a valuable tool for investigating visual cognition.

<h3 align="center">
Links: <a href="https://www.nature.com/articles/s42256-025-01130-7">NMI Paper</a> |<a href="https://www.tsinghua.edu.cn/info/1175/122677.htm">清华新闻Tsinghua News</a> | <a href="https://mp.weixin.qq.com/s/BmuptumV08AXry9V6hiLVg">清华自动化新闻Tsinghua DA News</a>
</h3>

## Usage

Please refer to [GET_STARTED.md](GET_STARTED.md) for training and evaluation instructions. 
Our pretrained model can be downloaded from [this link](https://drive.google.com/file/d/1OIPZyTozJatdd40VZKaJHsw0ShaujQ0Y/view?usp=sharing).

## Acknowledgements
This repository is built using the [timm](https://github.com/huggingface/pytorch-image-models) library and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) repository.

## Reference
If you find our code or papers useful for your research, please cite:
```
@article{wang2025emulating,
  title={Emulating human-like adaptive vision for efficient and flexible machine visual perception},
  author={Wang, Yulin and Yue, Yang and Yue, Yang and Wang, Huanqian and Jiang, Haojun and Han, Yizeng and Ni, Zanlin and Pu, Yifan and Shi, Minglei and Lu, Rui and others},
  journal={Nature Machine Intelligence},
  pages={1--19},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## Contact
If you have any question, feel free to contact the authors.

Yulin Wang(王语霖): yulin-wang@tsinghua.edu.cn

Yang Yue(乐洋): le-y22@mails.tsinghua.edu.cn

Yang Yue(乐阳): yueyang22@mails.tsinghua.edu.cn

