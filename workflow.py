"""Release integration helpers; original modules remain unchanged."""

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from metrics import boundary_counts, summarize


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((value.dtype, tuple(value.shape))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fit_head(model, loader, epochs=5, learning_rate=0.001, weight_decay=0.0001):
    """Fit a head on an externally supplied training-only episode loader."""
    if epochs < 1:
        raise ValueError("At least one epoch is required")
    if any(p.requires_grad for p in model.base_model.parameters()):
        raise ValueError("The base must be fully frozen")
    device = next(model.allocator.parameters()).device
    optimizer = torch.optim.AdamW(
        model.allocator.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    frozen = state_digest(model.base_model)
    history = []
    for epoch in range(epochs):
        model.train()
        if model.base_model.training:
            raise RuntimeError("The frozen base must remain in evaluation mode")
        total, batches = 0.0, 0
        for raw in loader:
            query, target, support, support_mask = [t.to(device) for t in raw[:4]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(support, support_mask, query)
            loss = F.cross_entropy(logits, target, ignore_index=255)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            gradients = [p.grad for p in model.allocator.parameters() if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise FloatingPointError("Missing or nonfinite head gradients")
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        if not batches:
            raise ValueError("Empty training loader")
        if state_digest(model.base_model) != frozen:
            raise RuntimeError("Frozen base tensors changed")
        history.append({"epoch": epoch + 1, "batches": batches, "loss": total / batches})
    model.eval()
    return history


def _class_index(value):
    while isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Evaluation requires one class index per episode")
        return int(value.item())
    return int(value)


def _row(prediction, target, class_index):
    valid = target != 255
    row = {
        "class_index": class_index,
        "tp": int(((prediction == 1) & (target == 1) & valid).sum()),
        "fp": int(((prediction == 1) & (target == 0) & valid).sum()),
        "fn": int(((prediction == 0) & (target == 1) & valid).sum()),
        "tn": int(((prediction == 0) & (target == 0) & valid).sum()),
    }
    row.update(boundary_counts(prediction, target))
    return row


def evaluate_pair(model, episodes, class_names):
    """Evaluate frozen base and ASRI on the exact same supplied episodes, per fold."""
    device = next(model.allocator.parameters()).device
    rows = {"SMNet": [], "ASRINet": []}
    model.eval()
    with torch.no_grad():
        for query, target, support, support_mask, class_id in episodes:
            if query.shape[0] != 1 or target.shape[0] != 1:
                raise ValueError("Evaluation requires batch size 1")
            index = _class_index(class_id)
            if not 0 <= index < len(class_names):
                raise ValueError("Fold-local class index out of range")
            query, support, support_mask = [t.to(device) for t in (query, support, support_mask)]
            ground_truth = target.detach().cpu().numpy()[0]
            base_logits, _ = model.base_model(support, support_mask, query, None)
            refined_logits = model(support, support_mask, query)
            for name, logits in (("SMNet", base_logits), ("ASRINet", refined_logits)):
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite output logits")
                prediction = logits.argmax(1).detach().cpu().numpy()[0]
                rows[name].append(_row(prediction, ground_truth, index))
    if not rows["SMNet"]:
        raise ValueError("Empty evaluation bank")
    return {name: {"metrics": summarize(values, class_names), "episodes": values}
            for name, values in rows.items()}


class GlobalOffset(nn.Module):
    """Matched-budget comparator: one zero-initialized bias on the base margin."""

    def __init__(self, base):
        super().__init__()
        self.base_model = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        self.beta = nn.Parameter(torch.zeros(()))

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, support, support_mask, query):
        with torch.no_grad():
            logits, _ = self.base_model(support, support_mask, query, None)
        margin = logits[:, 1:2] - logits[:, 0:1] + self.beta
        return torch.cat((-0.5 * margin, 0.5 * margin), dim=1)
