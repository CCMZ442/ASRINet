from pathlib import Path
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
from asri_model import ASRINet, block_mean
from bilinear_control import BilinearControlNet

VARIANTS = ('A1', 'A2', 'A3', 'A4', 'A5', 'A6')


class ControlNet(ASRINet):
    def __init__(self, base_model, variant):
        if variant not in ('A1','A2','A3','A4','A6'):
            raise ValueError('invalid two-branch control')
        modes = {'A2':'RAW_SHAPE_ONLY','A3':'PARENT_ONLY'}
        super().__init__(base_model, modes.get(variant, 'FREE'))
        self.variant = variant

    def _select_children(self, values):
        if self.variant == 'A1':
            return values['expanded_raw_bias']
        return super()._select_children(values)

    def forward(self, support, support_mask, query):
        if self.variant != 'A4':
            return super().forward(support, support_mask, query)
        if support.shape[1] != 1 or support_mask.shape[1] != 1:
            raise ValueError('C5 only trains/evaluates the approved1shot control')
        self._captures.clear()
        with torch.no_grad():
            logits, _ = self.base_model(support, support_mask, query, None)
        if len(self._captures) != 2:
            raise RuntimeError('support/query layer1 capture order differs')
        posterior = block_mean(torch.softmax(logits, dim=1)[:,1:2], self.scale)
        feature = self._captures[1].detach()
        if feature.shape[-2:] != posterior.shape[-2:]:
            feature = F.interpolate(feature, posterior.shape[-2:], mode='bilinear', align_corners=True)
        # Only head conditions are removed; the carrier and margin bypass stay intact.
        values = self.allocator(feature, torch.zeros_like(posterior))
        residual = F.pixel_shuffle(self._select_children(values), self.scale)
        if residual.shape[-2:] != logits.shape[-2:]:
            raise RuntimeError('residual/carrier geometry mismatch')
        margin = logits[:,1:2] - logits[:,0:1] + residual
        self.last_diagnostics = {key:values[key].detach() for key in ('raw_shape','raw_bias','uncertainty')}
        return torch.cat((-.5*margin,.5*margin), dim=1)


def build_control(base_model, variant):
    if variant not in VARIANTS:
        raise ValueError('invalid approved ablation variant')
    if variant == 'A5':
        return BilinearControlNet(base_model)
    return ControlNet(base_model, variant)
