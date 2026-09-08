# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for nanoACT: compact single-camera path and optional latent experiments.
# ResNet and frozen batch normalization adapt torchvision's BSD-3-Clause code;
# see licenses/torchvision-BSD-3-Clause.txt and THIRD_PARTY.md.
"""Single-camera ACT, adapted from LeRobot's policies/act/modeling_act.py.

This port keeps one observation, post-norm transformers and a ResNet-18 backbone.
It omits temporal ensembling, multiple cameras, environment-state inputs and
other backbone configurations. Transformer tensors are (sequence, batch, channel).
Model parameter names match LeRobot under the corresponding configuration;
checkpoint container formats differ.

During training, the VAE encoder reads the state and demonstrated action chunk
and supplies a sampled latent. At inference the latent is zero and the encoder
is not called. Optional flags support controlled changes to this training path.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

STATE_DIM = ACTION_DIM = 14  # ALOHA: 2 arms x (6 joints + gripper)


@dataclass
class ACTConfig:
    """Defaults follow LeRobot's ACT recipe."""

    # State/action dimensions default to ALOHA; model callers can override them.
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    chunk_size: int = 100
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1  # the original's config says 7, but its code only ran 1
    n_vae_encoder_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 10.0
    # False bypasses the encoder and KL term; its parameters remain in the model.
    use_vae: bool = True
    # Per-dimension floor on batch-mean KL, in nats. Below the floor the penalty
    # is constant; this does not guarantee informative latents or decoder use.
    free_bits: float = 0.0
    # With use_vae=False, use N(0, I) at training and zero at inference.
    noise_latent: bool = False
    # Give the VAE encoder detached backbone features pooled to 12 image tokens.
    # False preserves the standard model's parameter names and computations.
    encoder_sees_image: bool = False
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0
    batch_size: int = 8


# --------------------------------------------------------------------------------------
# Backbone: ResNet-18 up to layer4, vendored so it can be edited in place.
# --------------------------------------------------------------------------------------


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm with frozen statistics — the affine transform is a constant.

    ACT trains at batch size 8, far too small for stable batch statistics, so the
    ImageNet statistics are baked in and never updated. Registered as buffers, not
    parameters, which is why the backbone counts 11.17M trainable weights and not more.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:
        scale = self.weight.reshape(1, -1, 1, 1) * (self.running_var.reshape(1, -1, 1, 1) + self.eps).rsqrt()
        return x * scale + (self.bias.reshape(1, -1, 1, 1) - self.running_mean.reshape(1, -1, 1, 1) * scale)


def conv3x3(inp: int, out: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """The ResNet-18/34 residual block. Names match torchvision so weights load 1:1."""

    def __init__(self, inp: int, out: int, stride: int = 1):
        super().__init__()
        self.conv1, self.bn1 = conv3x3(inp, out, stride), FrozenBatchNorm2d(out)
        self.conv2, self.bn2 = conv3x3(out, out), FrozenBatchNorm2d(out)
        # Only needed when the block changes shape; torchvision names it `downsample`.
        self.downsample = (
            nn.Sequential(nn.Conv2d(inp, out, 1, stride=stride, bias=False), FrozenBatchNorm2d(out))
            if stride != 1 or inp != out
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(x)) + identity, inplace=True)


class ResNet18(nn.Module):
    """ResNet-18 truncated at layer4: (B, 3, 480, 640) -> (B, 512, 15, 20)."""

    out_channels = 512

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._layer(64, 64, stride=1)
        self.layer2 = self._layer(64, 128, stride=2)
        self.layer3 = self._layer(128, 256, stride=2)
        self.layer4 = self._layer(256, 512, stride=2)

    @staticmethod
    def _layer(inp: int, out: int, stride: int) -> nn.Sequential:
        return nn.Sequential(BasicBlock(inp, out, stride), BasicBlock(out, out))

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def load_imagenet_weights(self) -> None:
        """Load torchvision's IMAGENET1K_V1 weights; must be an exact match minus the head."""
        from torchvision.models import ResNet18_Weights

        sd = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, f"backbone weight mismatch: {missing} / {unexpected}"


# --------------------------------------------------------------------------------------
# Position embeddings
# --------------------------------------------------------------------------------------


def sinusoidal_pos_embedding(num_positions: int, dim: int) -> Tensor:
    """1D sinusoidal embedding ("Attention is All You Need"), for the VAE encoder input."""
    pos = torch.arange(num_positions, dtype=torch.float64).unsqueeze(1)
    # `2 * (i // 2)` pairs each sin with its cos at the same frequency.
    freq = torch.pow(10000, 2 * (torch.arange(dim, dtype=torch.float64) // 2) / dim)
    table = pos / freq
    table[:, 0::2] = table[:, 0::2].sin()
    table[:, 1::2] = table[:, 1::2].cos()
    return table.float()


class SinusoidalPosEmbedding2d(nn.Module):
    """2D sinusoidal embedding for the image feature map, as in DETR.

    Row/column indices are 1-based and normalized to [0, 2π]; both quirks are inherited
    from the original implementation and kept so the embedding matches value for value.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim  # half of dim_model: rows and columns each get dim channels

    def forward(self, x: Tensor) -> Tensor:
        ones = torch.ones_like(x[0, :1])  # (1, H, W)
        y_range = ones.cumsum(1) / (ones.shape[1] + 1e-6) * 2 * math.pi
        x_range = ones.cumsum(2) / (ones.shape[2] + 1e-6) * 2 * math.pi

        freq = 10000 ** (2 * (torch.arange(self.dim, dtype=torch.float32, device=x.device) // 2) / self.dim)
        y_range, x_range = y_range.unsqueeze(-1) / freq, x_range.unsqueeze(-1) / freq  # (1, H, W, dim)
        # Stack-then-flatten interleaves sine and cosine terms.
        embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        return torch.cat((embed_y, embed_x), dim=3).permute(0, 3, 1, 2)  # (1, 2*dim, H, W)


# --------------------------------------------------------------------------------------
# Transformer (post-norm, as the recipe uses)
# --------------------------------------------------------------------------------------


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.dim_model), nn.LayerNorm(cfg.dim_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1, self.dropout2 = nn.Dropout(cfg.dropout), nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, pos_embed: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        # Position is added to queries and keys but not values — the DETR convention.
        q = k = x + pos_embed
        x = self.norm1(x + self.dropout1(self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]))
        return self.norm2(x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x))))))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        self.multihead_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)
        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.norm3 = nn.LayerNorm(cfg.dim_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, memory: Tensor, dec_pos: Tensor, enc_pos: Tensor) -> Tensor:
        q = k = x + dec_pos
        x = self.norm1(x + self.dropout1(self.self_attn(q, k, value=x)[0]))
        cross = self.multihead_attn(query=x + dec_pos, key=memory + enc_pos, value=memory)[0]
        x = self.norm2(x + self.dropout2(cross))
        return self.norm3(x + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(x))))))


class Encoder(nn.Module):
    def __init__(self, cfg: ACTConfig, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(n_layers)])
        self.norm = nn.Identity()  # a LayerNorm only under pre-norm, which the recipe disables

    def forward(self, x: Tensor, pos_embed: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed, key_padding_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.dim_model)

    def forward(self, x: Tensor, memory: Tensor, dec_pos: Tensor, enc_pos: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, dec_pos, enc_pos)
        return self.norm(x)


# --------------------------------------------------------------------------------------
# ACT
# --------------------------------------------------------------------------------------


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.config = cfg

        # VAE encoder: [cls, state, *actions] -> latent distribution. Training only.
        self.vae_encoder = Encoder(cfg, cfg.n_vae_encoder_layers)
        self.vae_encoder_cls_embed = nn.Embedding(1, cfg.dim_model)
        self.vae_encoder_robot_state_input_proj = nn.Linear(cfg.state_dim, cfg.dim_model)
        self.vae_encoder_action_input_proj = nn.Linear(cfg.action_dim, cfg.dim_model)
        self.vae_encoder_latent_output_proj = nn.Linear(cfg.dim_model, cfg.latent_dim * 2)
        self.register_buffer(
            "vae_encoder_pos_enc", sinusoidal_pos_embedding(2 + cfg.chunk_size, cfg.dim_model).unsqueeze(0)
        )

        self.backbone = ResNet18()

        # Transformer. Encoder tokens are [latent, state, *image_feature_map_pixels].
        self.encoder = Encoder(cfg, cfg.n_encoder_layers)
        self.decoder = Decoder(cfg)
        self.encoder_robot_state_input_proj = nn.Linear(cfg.state_dim, cfg.dim_model)
        self.encoder_latent_input_proj = nn.Linear(cfg.latent_dim, cfg.dim_model)
        self.encoder_img_feat_input_proj = nn.Conv2d(ResNet18.out_channels, cfg.dim_model, kernel_size=1)
        self.encoder_1d_feature_pos_embed = nn.Embedding(2, cfg.dim_model)  # latent + state
        self.encoder_cam_feat_pos_embed = SinusoidalPosEmbedding2d(cfg.dim_model // 2)

        # Learned queries, one per action in the chunk (DETR object queries).
        self.decoder_pos_embed = nn.Embedding(cfg.chunk_size, cfg.dim_model)
        self.action_head = nn.Linear(cfg.dim_model, cfg.action_dim)

        # The VAE encoder's view of the image, when it has one. Created after every other
        # module so that under a given seed the rest of the model is initialized exactly as
        # it is with the flag off, and left at PyTorch's default init like the decoder-side
        # `encoder_img_feat_input_proj` (the Xavier loop below covers encoder and decoder
        # only). Not created at all with the flag off: the state_dict keys stay the shipped
        # ones, so existing checkpoints load unchanged and names still match LeRobot's.
        if cfg.encoder_sees_image:
            self.vae_encoder_img_input_proj = nn.Conv2d(ResNet18.out_channels, cfg.dim_model, kernel_size=1)
            self.vae_encoder_img_pos_embed = SinusoidalPosEmbedding2d(cfg.dim_model // 2)

        for p in [*self.encoder.parameters(), *self.decoder.parameters()]:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_latent(
        self,
        state: Tensor,
        actions: Tensor,
        action_is_pad: Tensor,
        image_feats: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compress the ground-truth chunk into latent parameters (mu, 2·log σ).

        `image_feats` is the decoder's backbone feature map (B, 512, 15, 20), required when
        `encoder_sees_image` is on and ignored when it is off.
        """
        batch = state.shape[0]
        tokens = torch.cat(
            [
                self.vae_encoder_cls_embed.weight.unsqueeze(0).expand(batch, -1, -1),
                self.vae_encoder_robot_state_input_proj(state).unsqueeze(1),
                self.vae_encoder_action_input_proj(actions),
            ],
            dim=1,
        )  # (B, chunk+2, D)
        # cls and state are never padding; padded actions must not inform the latent.
        mask = torch.cat([torch.zeros(batch, 2, dtype=torch.bool, device=state.device), action_is_pad], dim=1)
        pos = self.vae_encoder_pos_enc
        if self.config.encoder_sees_image:
            if image_feats is None:
                raise ValueError(
                    "encoder_sees_image is on: encode_latent needs the backbone feature map — "
                    "call ACT.posterior(image, state, actions, action_is_pad) instead"
                )
            # Pool to 12 tokens to limit the additional encoder computation.
            pooled = F.adaptive_avg_pool2d(image_feats, (3, 4))  # (B, 512, 3, 4)
            img_tokens = self.vae_encoder_img_input_proj(pooled).flatten(2).transpose(1, 2)  # (B, 12, D)
            tokens = torch.cat([tokens, img_tokens], dim=1)
            mask = torch.cat(
                [mask, torch.zeros(batch, img_tokens.shape[1], dtype=torch.bool, device=state.device)], dim=1
            )
            img_pos = self.vae_encoder_img_pos_embed(pooled).flatten(2).transpose(1, 2)  # (1, 12, D)
            pos = torch.cat([pos, img_pos.to(pos.dtype)], dim=1)
        cls_out = self.vae_encoder(tokens.transpose(0, 1), pos.transpose(0, 1), key_padding_mask=mask)[0]
        params = self.vae_encoder_latent_output_proj(cls_out)
        return params[:, : self.config.latent_dim], params[:, self.config.latent_dim :]

    def posterior(
        self, image: Tensor, state: Tensor, actions: Tensor, action_is_pad: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The latent distribution as training computes it.

        Whatever the flag, this is what `forward` feeds the reparameterization, so code
        that inspects the latent never has to know how the checkpoint was trained.
        """
        if not self.config.encoder_sees_image:
            return self.encode_latent(state, actions, action_is_pad)
        return self.encode_latent(state, actions, action_is_pad, image_feats=self.backbone(image).detach())

    def forward(
        self,
        image: Tensor,
        state: Tensor,
        actions: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Normalized image (B,3,H,W) and state (B,14) -> normalized action chunk (B,chunk,14)."""
        # With the flag on the encoder needs the feature map before the latent is drawn;
        # with it off the order of operations is exactly what it was, so the RNG draws
        # (reparameterization, dropout) fall in the same order.
        feature_map = self.backbone(image) if self.config.encoder_sees_image else None
        if self.config.use_vae and self.training and actions is not None:
            mu, log_sigma_x2 = self.encode_latent(
                state, actions, action_is_pad,
                image_feats=None if feature_map is None else feature_map.detach(),
            )
            latent = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)  # reparameterization
        else:
            mu = log_sigma_x2 = None
            latent = torch.zeros(state.shape[0], self.config.latent_dim, device=state.device)
            if self.config.noise_latent and self.training:
                latent = torch.randn_like(latent)  # noise at training, zeros at inference

        if feature_map is None:
            feature_map = self.backbone(image)  # (B, 512, 15, 20)
        cam_pos = self.encoder_cam_feat_pos_embed(feature_map).to(feature_map.dtype)
        cam_tokens = self.encoder_img_feat_input_proj(feature_map).flatten(2).permute(2, 0, 1)  # (HW, B, D)

        tokens = torch.cat(
            [
                torch.stack([self.encoder_latent_input_proj(latent), self.encoder_robot_state_input_proj(state)]),
                cam_tokens,
            ]
        )  # (2 + HW, B, D)
        pos = torch.cat(
            [self.encoder_1d_feature_pos_embed.weight.unsqueeze(1), cam_pos.flatten(2).permute(2, 0, 1)]
        )  # (2 + HW, 1, D)

        memory = self.encoder(tokens, pos)
        queries = torch.zeros(self.config.chunk_size, state.shape[0], self.config.dim_model, device=state.device)
        out = self.decoder(queries, memory, self.decoder_pos_embed.weight.unsqueeze(1), pos)
        return self.action_head(out.transpose(0, 1)), mu, log_sigma_x2


def compute_loss(
    model: ACT, image: Tensor, state: Tensor, actions: Tensor, action_is_pad: Tensor
) -> tuple[Tensor, float, float]:
    """L1 on unpadded actions + β·KL(latent || N(0, I))."""
    pred, mu, log_sigma_x2 = model(image, state, actions, action_is_pad)
    valid = ~action_is_pad.unsqueeze(-1)
    l1 = (F.l1_loss(actions, pred, reduction="none") * valid).sum() / (valid.sum() * actions.shape[-1]).clamp_min(1)
    if mu is None:  # CVAE ablated: no latent distribution, so nothing to regularize
        return l1, l1.item(), 0.0
    per_dim = -0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())  # (B, latent_dim)
    kl = per_dim.sum(-1).mean()  # reported unchanged, so runs stay comparable
    penalty = kl
    if model.config.free_bits > 0:
        # Floor each dimension's *batch-mean* KL, which is the original formulation: a
        # per-sample floor would let a dimension stay active for one sample and dead for the
        # rest, which is not the constraint we want.
        penalty = per_dim.mean(0).clamp_min(model.config.free_bits).sum()
    return l1 + model.config.kl_weight * penalty, l1.item(), kl.item()
