from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


MODES = ("FREE", "RAW_SHAPE_ONLY", "PARENT_ONLY")


def block_mean(value: torch.Tensor, scale: int) -> torch.Tensor:
    if value.shape[-2] % scale or value.shape[-1] % scale:
        raise ValueError("spatial dimensions must be divisible by scale")
    return F.avg_pool2d(value, kernel_size=scale, stride=scale)


def center_children(children: torch.Tensor, scale: int) -> torch.Tensor:
    if children.shape[1] != scale * scale:
        raise ValueError("child tensor must have scale squared channels")
    return children - children.mean(dim=1, keepdim=True)


class ASRIHead(nn.Module):
    """Predict subpixel residual allocation from shallow query detail and coarse posterior."""

    def __init__(self, channels: int = 256, hidden_channels: int = 32, scale: int = 4):
        super().__init__()
        if hidden_channels % 4:
            raise ValueError("hidden channels must be divisible by four")
        self.scale = scale
        self.query_projection = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
        )
        self.shape_body = nn.Sequential(
            nn.Conv2d(hidden_channels + 2, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
        )
        self.shape_output = nn.Conv2d(hidden_channels, scale * scale, 1)
        self.bias_body = nn.Sequential(
            nn.Conv2d(hidden_channels + 2, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
        )
        self.bias_output = nn.Conv2d(hidden_channels, 1, 1)
        self.gate_output = nn.Conv2d(hidden_channels, 1, 1)
        self.log_beta = nn.Parameter(torch.tensor(math.log(math.expm1(0.25))))
        for layer in (self.shape_output, self.bias_output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def forward(
        self,
        query: torch.Tensor,
        parent_probability: torch.Tensor,
        use_query_feature: bool = True,
    ) -> dict[str, torch.Tensor]:
        uncertainty = 4.0 * parent_probability * (1.0 - parent_probability)
        projected = self.query_projection(query)
        if not use_query_feature:
            projected = torch.zeros_like(projected)
        local_mean = F.avg_pool2d(projected, kernel_size=3, stride=1, padding=1)
        contrast = projected - local_mean

        shape_context = torch.cat([contrast, parent_probability, uncertainty], dim=1)
        raw_shape = self.shape_output(self.shape_body(shape_context))
        centered_shape = center_children(raw_shape, self.scale)

        bias_context = torch.cat([local_mean, parent_probability, uncertainty], dim=1)
        bias_feature = self.bias_body(bias_context)
        raw_bias = self.bias_output(bias_feature)
        reliability = torch.sigmoid(self.gate_output(bias_feature))
        beta = F.softplus(self.log_beta)
        controlled_bias = uncertainty * reliability * torch.tanh(raw_bias) * beta
        expanded_raw_bias = raw_bias.expand(-1, self.scale * self.scale, -1, -1)
        expanded_controlled_bias = controlled_bias.expand_as(expanded_raw_bias)
        return {
            "raw_shape": raw_shape,
            "centered_shape": centered_shape,
            "raw_bias": raw_bias,
            "controlled_bias": controlled_bias,
            "expanded_raw_bias": expanded_raw_bias,
            "expanded_controlled_bias": expanded_controlled_bias,
            "uncertainty": uncertainty,
            "reliability": reliability,
            "beta": beta,
        }


class ASRINet(nn.Module):
    """Frozen SMNet carrier with adaptive spatial residual inference."""

    def __init__(self, base_model: nn.Module, mode: str, scale: int = 4):
        super().__init__()
        if mode not in MODES:
            raise ValueError("invalid ASRI mode")
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        self.mode = mode
        self.scale = scale
        self.allocator = ASRIHead(scale=scale)
        self._captures: list[torch.Tensor] = []
        self._hook = self.base_model.layer1.register_forward_hook(
            lambda _module, _inputs, output: self._captures.append(output)
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def _select_children(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "FREE":
            return values["raw_shape"] + values["expanded_raw_bias"]
        if self.mode == "RAW_SHAPE_ONLY":
            return values["raw_shape"]
        if self.mode == "PARENT_ONLY":
            return values["raw_shape"] + values["expanded_raw_bias"]
        raise RuntimeError("unreachable ASRI mode")

    def forward(self, support, support_mask, query):
        if support.shape[1] != 1 or support_mask.shape[1] != 1:
            raise ValueError("ASRINet D1 supports exactly one support shot")
        self._captures.clear()
        with torch.no_grad():
            logits, _ = self.base_model(support, support_mask, query, None)
        if len(self._captures) != 2:
            raise RuntimeError("expected one support and one query layer1 capture")
        query_feature = self._captures[1].detach()
        foreground_probability = torch.softmax(logits, dim=1)[:, 1:2]
        parent_probability = block_mean(foreground_probability, self.scale)
        if query_feature.shape[-2:] != parent_probability.shape[-2:]:
            query_feature = F.interpolate(
                query_feature,
                parent_probability.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        values = self.allocator(
            query_feature,
            parent_probability,
            use_query_feature=self.mode != "PARENT_ONLY",
        )
        residual = F.pixel_shuffle(self._select_children(values), self.scale)
        if residual.shape[-2:] != logits.shape[-2:]:
            raise RuntimeError("allocation residual and base logits have different geometry")
        base_margin = logits[:, 1:2] - logits[:, 0:1]
        final_margin = base_margin + residual
        self.last_diagnostics = {
            key: value.detach()
            for key, value in values.items()
            if key in ("raw_shape", "raw_bias", "uncertainty")
        }
        return torch.cat((-0.5 * final_margin, 0.5 * final_margin), dim=1)

    def close(self):
        self._hook.remove()
