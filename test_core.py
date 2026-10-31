"""Synthetic CPU interface tests, not benchmark performance evaluation."""

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from asri_model import ASRIHead, block_mean
from controls import build_control
from multishot import MultiShotASRI
from simple_head import SimpleHead
from workflow import GlobalOffset, evaluate_pair, fit_head, state_digest


class SyntheticBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 256, kernel_size=4, stride=4)

    def forward(self, support, support_mask, query, target=None):
        for index in range(support.shape[1]):
            self.layer1(support[:, index])
        feature = self.layer1(query)
        margin = F.interpolate(feature.mean(1, keepdim=True), query.shape[-2:],
                               mode="bilinear", align_corners=True)
        return torch.cat((-0.5 * margin, 0.5 * margin), 1), None


def inputs(shot=1, batch=1):
    return (torch.randn(batch, shot, 3, 32, 32),
            torch.randint(0, 2, (batch, shot, 32, 32)),
            torch.randn(batch, 3, 32, 32))


def tearDownModule():
    if torch.cuda.is_initialized():
        raise AssertionError("CPU release tests initialized CUDA")


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3407)

    def test_parameter_counts(self):
        head = ASRIHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 28563)
        active = sum(p.numel() for n, p in head.named_parameters()
                     if not n.startswith(("gate_output.", "log_beta")))
        self.assertEqual(active, 28529)
        simple = SimpleHead(head.query_projection)
        self.assertEqual(sum(p.numel() for p in simple.parameters()), 27856)

    def test_zero_output_and_no_diagnostic_gradient(self):
        head = ASRIHead()
        values = head(torch.randn(2, 256, 8, 8), torch.rand(2, 1, 8, 8))
        residual = values["raw_shape"] + values["expanded_raw_bias"]
        self.assertEqual(int(torch.count_nonzero(residual)), 0)
        residual.square().sum().backward()
        self.assertIsNone(head.gate_output.weight.grad)
        self.assertIsNone(head.log_beta.grad)

    def test_zero_identity_one_and_five_shot(self):
        for shot in (1, 5):
            model = MultiShotASRI(SyntheticBase(), "FREE")
            try:
                support, mask, query = inputs(shot)
                original = model.base_model(support, mask, query, None)[0]
                refined = model(support, mask, query)
                self.assertEqual(tuple(refined.shape), (1, 2, 32, 32))
                torch.testing.assert_close(original.softmax(1), refined.softmax(1))
            finally:
                model.close()

    def test_frozen_base_and_real_head_update(self):
        model = MultiShotASRI(SyntheticBase(), "FREE")
        try:
            support, mask, query = inputs(batch=2)
            target = torch.ones(2, 32, 32, dtype=torch.long)
            frozen = state_digest(model.base_model)
            original_head = state_digest(model.allocator)
            history = fit_head(model, [(query, target, support, mask)], epochs=2)
            self.assertEqual(len(history), 2)
            self.assertEqual(frozen, state_digest(model.base_model))
            self.assertNotEqual(original_head, state_digest(model.allocator))
            self.assertFalse(model.base_model.training)
        finally:
            model.close()

    def test_actual_free_gradient_coverage(self):
        head = ASRIHead()
        values = head(torch.randn(2, 256, 8, 8), torch.rand(2, 1, 8, 8))
        (values["raw_shape"] + values["expanded_raw_bias"]).sum().backward()
        names = {name for name, parameter in head.named_parameters() if parameter.grad is not None}
        self.assertEqual(sum(p.numel() for name, p in head.named_parameters() if name in names), 28529)

    def test_pixelshuffle_channel_order_and_block_mean(self):
        channels = torch.arange(16.0).reshape(1, 16, 1, 1)
        output = F.pixel_shuffle(channels, 4)
        torch.testing.assert_close(output.reshape(4, 4), torch.arange(16.0).reshape(4, 4))
        shifted = F.pixel_shuffle(channels + 0.7, 4)
        torch.testing.assert_close(shifted - output, torch.full_like(output, 0.7))
        self.assertAlmostEqual(float(block_mean(output, 4)), 7.5)

    def test_archived_ablation_interfaces(self):
        support, mask, query = inputs()
        for variant in ("A1", "A2", "A3", "A4", "A5", "A6"):
            model = build_control(SyntheticBase(), variant)
            try:
                self.assertEqual(tuple(model(support, mask, query).shape), (1, 2, 32, 32))
            finally:
                model.close()

    def test_simple_head_zero_identity(self):
        model = MultiShotASRI(SyntheticBase(), "FREE")
        try:
            model.allocator = SimpleHead(model.allocator.query_projection)
            support, mask, query = inputs(5)
            a = model.base_model(support, mask, query, None)[0]
            b = model(support, mask, query)
            torch.testing.assert_close(a, b)
        finally:
            model.close()

    def test_paired_metrics_and_ignore(self):
        model = MultiShotASRI(SyntheticBase(), "FREE")
        try:
            support, mask, query = inputs()
            target = torch.zeros(1, 32, 32, dtype=torch.long)
            target[:, 8:24, 8:24] = 1
            target[:, 0, 0] = 255
            results = evaluate_pair(model, [(query, target, support, mask, 0)], ["defect"])
            self.assertEqual(results["SMNet"], results["ASRINet"])
            row = results["SMNet"]["episodes"][0]
            self.assertEqual(sum(row[k] for k in ("tp", "fp", "fn", "tn")), 32 * 32 - 1)
        finally:
            model.close()

    def test_checkpoint_roundtrip(self):
        model = MultiShotASRI(SyntheticBase(), "FREE")
        replica = MultiShotASRI(SyntheticBase(), "FREE")
        try:
            replica.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
            support, mask, query = inputs(5)
            torch.testing.assert_close(model(support, mask, query), replica(support, mask, query))
        finally:
            model.close()
            replica.close()

    def test_scalar_identity_and_gradient(self):
        model = GlobalOffset(SyntheticBase())
        support, mask, query = inputs()
        a = model.base_model(support, mask, query, None)[0]
        b = model(support, mask, query)
        torch.testing.assert_close(a.softmax(1), b.softmax(1))
        F.cross_entropy(b, torch.ones(1, 32, 32, dtype=torch.long)).backward()
        self.assertIsNotNone(model.beta.grad)
        self.assertTrue(bool(torch.isfinite(model.beta.grad)))
        self.assertTrue(all(p.grad is None for p in model.base_model.parameters()))

    def test_invalid_shot_and_grid(self):
        model = MultiShotASRI(SyntheticBase(), "FREE")
        try:
            with self.assertRaises(ValueError):
                model(*inputs(2))
            with self.assertRaises(ValueError):
                block_mean(torch.zeros(1, 1, 31, 32), 4)
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main()
