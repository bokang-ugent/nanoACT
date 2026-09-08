# Upstream code and licenses

## LeRobot ACT

`nanoact/model.py` adapts `src/lerobot/policies/act/modeling_act.py`.
The local reference used for extraction and verification is LeRobot 0.6.0.

- Source: <https://github.com/huggingface/lerobot>
- Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
- License: Apache-2.0; the upstream license is reproduced in
  [`licenses/LeRobot-APACHE-2.0.txt`](licenses/LeRobot-APACHE-2.0.txt).

The port keeps a single RGB camera, one observation, post-norm transformers,
and a ResNet-18 backbone. It removes configuration branches and supplies a
small training/data interface. Optional training variants bypass the VAE
encoder, substitute noise, change the KL penalty, or add image features to
the VAE encoder. Model parameter names are retained where the configurations
match. These are modifications to LeRobot, not an upstream release.

ACT's original implementation is <https://github.com/tonyzhaozh/act>
(MIT, Copyright (c) 2023 Tony Z. Zhao). LeRobot's ACT module credits Tony Z.
Zhao as a copyright holder, and LeRobot's license file states that parts of
its code derive from ALOHA (MIT, Tony Z. Zhao) and DETR (Apache-2.0,
Facebook, Inc.). Those notices are reproduced inside
[`licenses/LeRobot-APACHE-2.0.txt`](licenses/LeRobot-APACHE-2.0.txt) and
apply to the adapted model code. The original ACT training code is not
bundled in this repository.

## torchvision

The ResNet-18, residual block, and frozen batch normalization definitions in
`nanoact/model.py` adapt torchvision's implementations. The backbone omits
the classification head and exposes the last spatial feature map.

- Source: <https://github.com/pytorch/vision>
- Local reference version: torchvision 0.26.0.
- Copyright (c) Soumith Chintala 2016. All rights reserved.
- License: BSD-3-Clause, reproduced in
  [`licenses/torchvision-BSD-3-Clause.txt`](licenses/torchvision-BSD-3-Clause.txt).

ImageNet weights are downloaded through torchvision when training starts;
they are not distributed in this repository. Dataset files are downloaded
from their pinned Hugging Face repositories and are not distributed here.
