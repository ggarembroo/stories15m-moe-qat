"""Ternary / low-bit quantization-aware training for the 36M Mixtral MoE.

The module keeps the original expert weights as FP32 parameters and applies a
straight-through estimator around symmetric group quantization.  It supports
learnable per-group scales, outlier protection, calibration clipping, optional
block-Hadamard rotation and mixed-precision layers.

Tested with transformers 4.36+ and 5.x on ggml-org/stories15M_MOE.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def block_hadamard(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Orthogonal block transform along the last dimension."""
    if x.shape[-1] % block:
        raise ValueError(f"last dim {x.shape[-1]} is not divisible by block {block}")
    orig = x.shape
    y = x.reshape(-1, block).clone()
    h = 1
    while h < block:
        y = y.reshape(-1, 2, h)
        a = y[:, 0, :].clone()
        b = y[:, 1, :].clone()
        y[:, 0, :] = a + b
        y[:, 1, :] = a - b
        y = y.reshape(-1, block)
        h *= 2
    return (y / math.sqrt(block)).reshape(orig)


class QATExperts(nn.Module):
    """Drop-in replacement for MixtralExperts with fake quantization.

    Parameters
    ----------
    orig:
        The original ``MixtralExperts`` module.
    bits:
        1 for ternary, 2/4 for symmetric integer quantization.
    group:
        Number of weights sharing one scale along the input dimension.
    scale_mode:
        ``group``, ``row`` or ``col``.
    learnable_scales:
        If true, log-scales are trained.  Otherwise they are fixed buffers.
    outlier_frac:
        Fraction of largest-magnitude weights kept in FP32 (0 disables).
    clip_percentile:
        Percentile used for scale initialization (1.0 means max).
    rotate:
        Apply block-Hadamard before quantization.
    block:
        Hadamard block size.
    """

    def __init__(
        self,
        orig: nn.Module,
        bits: int = 1,
        group: int = 32,
        scale_mode: str = "group",
        learnable_scales: bool = True,
        outlier_frac: float = 0.0,
        clip_percentile: float = 1.0,
        rotate: bool = False,
        block: int = 32,
    ) -> None:
        super().__init__()
        if scale_mode not in {"group", "row", "col"}:
            raise ValueError(f"unknown scale_mode {scale_mode!r}")
        if group <= 0:
            raise ValueError("group must be positive")
        if bits not in {1, 2, 4}:
            raise ValueError("bits must be 1, 2 or 4")

        self.num_experts = orig.num_experts
        self.hidden_dim = orig.hidden_dim
        self.intermediate_dim = orig.intermediate_dim
        self.act_fn = orig.act_fn
        self.bits = bits
        self.group = group
        self.scale_mode = scale_mode
        self.clip_percentile = clip_percentile
        self.outlier_frac = outlier_frac
        self.rotate = rotate
        self.block = block

        self.gate_up_proj = nn.Parameter(orig.gate_up_proj.detach().clone())
        self.down_proj = nn.Parameter(orig.down_proj.detach().clone())

        gate_scale = self._init_log_scale(self.gate_up_proj)
        down_scale = self._init_log_scale(self.down_proj)
        if learnable_scales:
            self.gate_log_scale = nn.Parameter(gate_scale)
            self.down_log_scale = nn.Parameter(down_scale)
        else:
            self.register_buffer("gate_log_scale", gate_scale)
            self.register_buffer("down_log_scale", down_scale)

        gate_mask = self._outlier_mask(self.gate_up_proj) if outlier_frac > 0 else None
        down_mask = self._outlier_mask(self.down_proj) if outlier_frac > 0 else None
        self.register_buffer("gate_mask", gate_mask)
        self.register_buffer("down_mask", down_mask)

    def _init_log_scale(self, w: torch.Tensor) -> torch.Tensor:
        e, rows, cols = w.shape
        wf = w.detach().float()
        p = self.clip_percentile
        if self.scale_mode == "group":
            if cols % self.group:
                raise ValueError(f"cols {cols} is not divisible by group {self.group}")
            blocks = wf.reshape(e, rows, cols // self.group, self.group)
            scale = torch.quantile(blocks.abs(), p, dim=-1)
        elif self.scale_mode == "row":
            scale = torch.quantile(wf.abs(), p, dim=2, keepdim=True)
        else:
            scale = torch.quantile(wf.abs(), p, dim=1, keepdim=True)
        return torch.log(scale.clamp_min(1e-4))

    def _outlier_mask(self, w: torch.Tensor) -> torch.Tensor:
        wf = w.detach().abs().float()
        threshold = torch.quantile(wf.flatten(), 1.0 - self.outlier_frac)
        return wf >= threshold

    def _quant(self, w: torch.Tensor, log_scale: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        e, rows, cols = w.shape
        if self.scale_mode == "group":
            blocks = w.reshape(e, rows, cols // self.group, self.group)
            scale = torch.exp(log_scale.clamp(-12, 4)).unsqueeze(-1)
        elif self.scale_mode == "row":
            blocks = w
            scale = torch.exp(log_scale.clamp(-12, 4))
        else:
            blocks = w
            scale = torch.exp(log_scale.clamp(-12, 4))

        if self.bits == 1:
            q = torch.clamp(torch.round(blocks / scale), -1, 1) * scale
        else:
            qmax = 2 ** (self.bits - 1) - 1
            q = torch.clamp(torch.round(blocks / scale), -qmax - 1, qmax) * scale

        q = q.reshape(e, rows, cols).to(w.dtype)
        if mask is not None:
            q = torch.where(mask, w, q)
        return (w + (q - w).detach()).to(w.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        wg = self.gate_up_proj
        wd = self.down_proj
        if self.rotate:
            wg = block_hadamard(wg, self.block)
            wd = block_hadamard(wd, self.block)
        wg = self._quant(wg, self.gate_log_scale, self.gate_mask)
        wd = self._quant(wd, self.down_log_scale, self.down_mask)

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x = hidden_states[token_idx]
            if self.rotate:
                x = block_hadamard(x, self.block)
            gate, up = F.linear(x, wg[expert_idx]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            if self.rotate:
                h = block_hadamard(h, self.block)
            y = F.linear(h, wd[expert_idx])
            y = y * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, y.to(final.dtype))
        return final


def replace_experts(
    model: nn.Module,
    bits: int = 1,
    group: int = 32,
    scale_mode: str = "group",
    learnable_scales: bool = True,
    outlier_frac: float = 0.0,
    clip_percentile: float = 1.0,
    rotate: bool = False,
    block: int = 32,
    skip_layers: Sequence[int] = (),
) -> nn.Module:
    """Replace routed experts in every layer except ``skip_layers``."""
    skip = set(skip_layers)
    for i, layer in enumerate(model.model.layers):
        if i in skip:
            continue
        layer.mlp.experts = QATExperts(
            layer.mlp.experts,
            bits=bits,
            group=group,
            scale_mode=scale_mode,
            learnable_scales=learnable_scales,
            outlier_frac=outlier_frac,
            clip_percentile=clip_percentile,
            rotate=rotate,
            block=block,
        )
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
