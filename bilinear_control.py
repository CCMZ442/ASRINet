"""Conventional shallow-query refinement control for ASRINet."""

import os
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

D1_ROOT = Path(os.environ.get(
    "ASRI_D1_ROOT", str(Path(__file__).resolve().parent.parent / "ASRINET_D1_ATTRIBUTION_20260923")
))
sys.path.insert(0, str(D1_ROOT))
from asri_model import ASRINet, block_mean


class BilinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.GroupNorm(4, 32), nn.GELU()
        )
        self.body = nn.Sequential(
            nn.Conv2d(34, 32, 3, padding=1, bias=False), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.GroupNorm(4, 32), nn.GELU(),
        )
        self.output = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, posterior):
        projected = self.query_projection(query)
        uncertainty = 4 * posterior * (1 - posterior)
        return self.output(self.body(torch.cat((projected, posterior, uncertainty), dim=1)))


class BilinearControlNet(ASRINet):
    def __init__(self, base_model):
        super().__init__(base_model, "FREE", scale=4)
        reference_projection = self.allocator.query_projection.state_dict()
        # Match FREE's post-construction RNG state so loader sampling is unchanged.
        with torch.random.fork_rng(devices=[]):
            control = BilinearHead()
        control.query_projection.load_state_dict(reference_projection)
        self.allocator = control
        self.refinement_enabled = True

    def forward(self, support, support_mask, query):
        if support.shape[1] != 1 or support_mask.shape[1] != 1:
            raise ValueError("D2 supports exactly one support shot")
        self._captures.clear()
        with torch.no_grad():
            logits, _ = self.base_model(support, support_mask, query, None)
        if len(self._captures) != 2:
            raise RuntimeError("expected support then query layer1 captures")
        margin = logits[:, 1:2] - logits[:, 0:1]
        if self.refinement_enabled:
            posterior = block_mean(torch.softmax(logits, dim=1)[:, 1:2], self.scale)
            feature = self._captures[1].detach()
            if feature.shape[-2:] != posterior.shape[-2:]:
                feature = F.interpolate(feature, posterior.shape[-2:], mode="bilinear", align_corners=True)
            coarse = self.allocator(feature, posterior)
            margin = margin + F.interpolate(coarse, logits.shape[-2:], mode="bilinear", align_corners=True)
        return torch.cat((-0.5 * margin, 0.5 * margin), dim=1)
