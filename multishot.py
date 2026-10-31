"""Native support-shot interface extension; ASRI computations and weights unchanged."""
import torch
import torch.nn.functional as F
from asri_model import ASRINet, block_mean


class MultiShotASRI(ASRINet):
    enabled = True

    def forward(self, support, support_mask, query):
        if support.ndim != 5 or support_mask.ndim != 4:
            raise ValueError('expected B,K,C,H,W supports and B,K,H,W masks')
        shots = support.shape[1]
        if shots not in (1, 5) or support.shape[:2] != support_mask.shape[:2]:
            raise ValueError('only matched native 1/5-shot inputs are supported')
        self._captures.clear()
        with torch.no_grad():
            logits, _ = self.base_model(support, support_mask, query, None)
        # The verified native SMNet loops over supports, then encodes the query.
        if len(self._captures) != shots + 1:
            raise RuntimeError('native SMNet support/query capture contract changed')
        if not self.enabled:
            return logits
        query_feature = self._captures[-1].detach()
        parent_probability = block_mean(torch.softmax(logits, dim=1)[:, 1:2], self.scale)
        if query_feature.shape[-2:] != parent_probability.shape[-2:]:
            query_feature = F.interpolate(query_feature, parent_probability.shape[-2:],
                                          mode='bilinear', align_corners=True)
        values = self.allocator(query_feature, parent_probability,
                                use_query_feature=self.mode != 'PARENT_ONLY')
        residual = F.pixel_shuffle(self._select_children(values), self.scale)
        if residual.shape[-2:] != logits.shape[-2:]:
            raise RuntimeError('allocation geometry mismatch')
        final_margin = logits[:, 1:2] - logits[:, 0:1] + residual
        self.last_diagnostics = {k: v.detach() for k, v in values.items()
                                 if k in ('raw_shape', 'raw_bias', 'uncertainty')}
        return torch.cat((-0.5 * final_margin, 0.5 * final_margin), dim=1)
