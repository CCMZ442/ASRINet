"""Same input information and output parameterization, without two branches."""
import copy
import torch
from torch import nn
import torch.nn.functional as F


class SimpleHead(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.query_projection = copy.deepcopy(projection)
        self.body = nn.Sequential(nn.Conv2d(66, 32, 3, padding=1, bias=False),
                                  nn.GroupNorm(4, 32), nn.GELU())
        self.output = nn.Conv2d(32, 16, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, parent_probability, use_query_feature=True):
        f = self.query_projection(query)
        if not use_query_feature:
            f = torch.zeros_like(f)
        local = F.avg_pool2d(f, 3, 1, 1)
        uncertainty = 4 * parent_probability * (1-parent_probability)
        shape = self.output(self.body(torch.cat((f-local, local, parent_probability, uncertainty), 1)))
        bias = torch.zeros_like(parent_probability)
        return dict(raw_shape=shape, raw_bias=bias,
                    expanded_raw_bias=bias.expand_as(shape), uncertainty=uncertainty)
