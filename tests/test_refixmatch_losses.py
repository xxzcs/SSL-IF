import unittest

import torch
import torch.nn.functional as F

from semilearn.algorithms.refixmatch.refixmatch import ReFixMatch


class _LossHarness:
    T = 0.5
    compute_refixmatch_unsup_loss = ReFixMatch.compute_refixmatch_unsup_loss

    @staticmethod
    def consistency_loss(logits, targets, name='ce', mask=None):
        loss = F.cross_entropy(logits, targets, reduction='none')
        return (loss * mask).mean()


class ReFixMatchLossTest(unittest.TestCase):
    def test_high_and_low_confidence_masks_are_complementary(self):
        model = _LossHarness()
        weak = torch.tensor([[5.0, 0.0], [0.2, 0.0]])
        strong = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        pseudo = torch.tensor([0, 0])
        high_mask = torch.tensor([1.0, 0.0])

        hard, soft = model.compute_refixmatch_unsup_loss(
            weak, strong, pseudo, high_mask)

        expected_hard = F.cross_entropy(strong[:1], pseudo[:1]) / 2
        target = F.softmax(weak[1:] / model.T, dim=-1)
        expected_soft = F.kl_div(
            F.log_softmax(strong[1:], dim=-1), target, reduction='sum') / 2
        self.assertTrue(torch.allclose(hard, expected_hard))
        self.assertTrue(torch.allclose(soft, expected_soft))

    def test_all_high_confidence_has_zero_soft_loss(self):
        model = _LossHarness()
        weak = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        strong = weak.clone().requires_grad_(True)
        hard, soft = model.compute_refixmatch_unsup_loss(
            weak, strong, torch.tensor([0, 1]), torch.ones(2))
        self.assertGreater(hard.item(), 0.0)
        self.assertEqual(soft.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
