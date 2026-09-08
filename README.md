# nanoACT

A compact adaptation of [LeRobot's](https://github.com/huggingface/lerobot)
implementation of **Action Chunking with Transformers (ACT)**.

nanoACT was written for our re-run of ACT's CVAE ablation
([act-cvae-forensics](https://github.com/aida-ugent/act-cvae-forensics)),
which needed controlled changes to the training path and measurements of
the latent. Beyond that study, it is meant for learning how ACT works and
for testing ideas about efficiency and policy performance. The simplified
code base makes it easier to follow the model and training path, change
one component, and evaluate the effect.

The model code is adapted directly from LeRobot's
`src/lerobot/policies/act/modeling_act.py`. nanoACT removes configuration
branches, keeps LeRobot's model parameter names, and adds a small
training/data interface and optional latent experiments. The model lives in
one file, [`nanoact/model.py`](nanoact/model.py), so its computations and
training variants are easy to inspect. See [THIRD_PARTY.md](THIRD_PARTY.md)
for the upstream sources and modifications.

nanoACT keeps the single-camera ACT configuration used in the study: one RGB
image, robot state, a ResNet-18 backbone, and a predicted action chunk. It
includes a training loop, pinned ALOHA demonstration downloads, decoded data
caches, and checkpoint inference. It does not implement LeRobot's full
configuration surface, multiple cameras, or temporal ensembling, and it
contains no simulator or evaluation loop.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra data
```

The `data` extra installs the download and video preparation dependencies.
Video decoding also needs a system FFmpeg installation supported by
[TorchCodec](https://github.com/pytorch/torchcodec#installing-torchcodec).
On macOS, `brew install ffmpeg` provides it. On Ubuntu, use
`sudo apt-get install ffmpeg`.

To use the model or an existing checkpoint without preparing data:

```bash
uv sync --frozen
```

The lockfile pins the dependency versions. The uv configuration selects
PyTorch's CUDA 12.6 wheels on Linux. GPU training needs a CUDA-compatible
driver; `--device cpu` runs training on the CPU.

## Train on ALOHA demonstrations

Four pinned datasets are supported: `transfer_cube_human`,
`transfer_cube_scripted`, `insertion_human`, and `insertion_scripted`.
Run these commands from this repository. Data goes into `./data`; set
`NANOACT_DATA_DIR` to an absolute path to use another location. Use the same
location for all three commands.

```bash
uv run --frozen --extra data python -m nanoact.download --dataset transfer_cube_human
uv run --frozen --extra data python -m nanoact.prepare_data --dataset transfer_cube_human
uv run --frozen python -m nanoact.train \
  --dataset transfer_cube_human --steps 100000 --seed 1000 \
  --out outputs/transfer_cube_human
```

The downloader fetches a fixed Hugging Face revision and records file
checksums. Preparation decodes the video once and keeps the dataset's
normalization statistics. The decoded RGB cache takes roughly 18 GB for a
20,000-frame dataset and 23 GB for `insertion_human`; check available disk
space first. These commands expect the pinned data layout; they are not a
general LeRobot dataset loader.

The default training recipe uses the VAE encoder, a KL weight of 10, and a
constant learning rate. Add `--no-vae` to bypass the encoder during
training; this saves its computation but keeps its parameters in the
checkpoint. `--kl-weight`, `--free-bits`, `--noise-latent`, and
`--encoder-sees-image` provide controlled training variants. `--grad-trace`
writes per-step gradient norms, `--clip-scope` changes how gradient clipping
is grouped, and `--mp4` decodes video during training instead of reading the
cache. See `python -m nanoact.train --help`.

Checkpoints contain model configuration, weights, normalization statistics,
dataset identity and revision, training seed, completed step count, and
source-file hashes. The hashes identify the code even before it is committed.
Checkpoints do not contain optimizer state, so training cannot be resumed
from them.

## Run a checkpoint

```python
import numpy as np
from nanoact.policy import NanoACT

policy = NanoACT("outputs/transfer_cube_human/step_100000.pt")
policy.reset()  # call at the start of each episode

# Replace these arrays with the environment's RGB image and robot state.
observation = {
    "pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)},
    "agent_pos": np.zeros(14, dtype=np.float32),
}
action = policy.select_action(observation)
```

The wrapper normalizes observations and returns actions in the dataset's
original action units. It predicts a chunk, executes it one action at a
time, then replans. The latent is zero at inference and the VAE encoder is
not called. Load checkpoints you trust: this wrapper uses PyTorch's pickle
checkpoint format.

The observation wrapper and the bundled training commands target ALOHA;
model callers can set other state and action dimensions through
`ACTConfig`. Rollout evaluation, latent probes, results, and figure
generation for the study live in the companion repository,
[act-cvae-forensics](https://github.com/aida-ugent/act-cvae-forensics).

## Verify the package

```bash
uv run --frozen --extra data pytest
uv build
```

The tests cover episode boundaries and padding, normalization, dataset
identity checks, a small training/checkpoint round trip, and inference
queue reset. They use synthetic data and do not download demonstrations
or pretrained weights.

We also checked the port against LeRobot 0.6.0 under the matched
configuration: parameter names and shapes, a synchronized forward pass and
loss, and one optimizer step agree. These checks do not establish
equivalence for every LeRobot configuration or for an entire training run.
LeRobot is not a runtime dependency. Under that configuration the parameter
names match, so weights can be copied between the two models; the
checkpoint file formats differ.

## Attribution and license

ACT was introduced by Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea
Finn in [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://www.roboticsproceedings.org/rss19/p016.html)
(RSS 2023). nanoACT adapts LeRobot's ACT implementation and torchvision's
ResNet and frozen batch normalization components.

See [THIRD_PARTY.md](THIRD_PARTY.md) for source attribution and modifications.
nanoACT's own additions are Copyright 2026 Bo Kang (Ghent University).
The project uses Apache-2.0, with the adapted torchvision components also
subject to their BSD-3-Clause terms. Dependency and dataset licenses remain
with their respective projects.

## Citing

If you use nanoACT, please cite ACT and LeRobot:

```bibtex
@inproceedings{zhao2023act,
  author    = {Tony Z. Zhao and Vikash Kumar and Sergey Levine and Chelsea Finn},
  title     = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2023},
  doi       = {10.15607/RSS.2023.XIX.016}
}

@misc{cadene2024lerobot,
  author = {R{\'e}mi Cadene and Simon Alibert and Alexander Soare and Quentin Gallou{\'e}dec and Adil Zouitine and Steven Palma and Pepijn Kooijmans and Michel Aractingi and Mustafa Shukor and Dana Aubakirova and Martino Russi and Francesco Capuano and Caroline Pascal and Jade Choghari and Khalil Meftah and Maxime Ellerbach and Jess Moss and Thomas Wolf},
  title  = {{LeRobot}: State-of-the-art Machine Learning for Real-World Robotics in {PyTorch}},
  howpublished = {\url{https://github.com/huggingface/lerobot}},
  year   = {2024}
}
```

If you use the training variants or the latent measurements, please also
cite the study nanoACT was written for:

```bibtex
@misc{kang2026latent,
  author        = {Bo Kang},
  title         = {The Latent That Never Was: A Forensic Re-run of the {CVAE} Ablation in Action Chunking Transformers},
  year          = {2026},
  eprint        = {2609.16745},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.16745}
}
```
