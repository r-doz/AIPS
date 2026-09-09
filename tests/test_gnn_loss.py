import unittest
import torch
from src.models.gnn import GNNTrainingLoss, poisson_nll


class GNNLossTests(unittest.TestCase):
    def test_original_exact_and_combined_formula(self):
        y = torch.tensor([[2., 8.], [6., 4.]])
        rates = torch.tensor([[1., 7.], [5., 3.]], requires_grad=True)
        self.assertTrue(torch.equal(GNNTrainingLoss(y)(rates, y), poisson_nll(rates, y)))
        combined = GNNTrainingLoss(y, 'combined', 2)
        torch.testing.assert_close(combined(rates, y), poisson_nll(rates, y)/5 + 2*torch.tensor(0.2))
        combined(rates, y).backward()
        self.assertTrue(torch.isfinite(rates.grad).all())

    def test_daily_only_ignores_spatial_redistribution(self):
        y = torch.tensor([[2., 8.], [6., 4.]])
        loss = GNNTrainingLoss(y, 'daily_only')
        redistributed = torch.tensor([[8., 2.], [4., 6.]])
        self.assertEqual(loss(redistributed, y).item(), 0)
        self.assertNotEqual(poisson_nll(redistributed, y).item(), poisson_nll(y, y).item())
        rates = torch.tensor([[1., 7.], [8., 4.]], requires_grad=True)
        loss(rates, y).backward()
        torch.testing.assert_close(rates.grad, torch.tensor([[-.05, -.05], [.05, .05]]))

    def test_scales_fixed_and_zero_safe(self):
        y = torch.zeros((2, 3))
        loss = GNNTrainingLoss(y, 'combined')
        loss(torch.ones_like(y), y+100)
        self.assertEqual(loss.daily_scale.item(), 1)
        self.assertEqual(loss.cell_scale.item(), 1)
        self.assertTrue(torch.isfinite(loss(torch.ones_like(y), y)))
        with self.assertRaises(ValueError):
            GNNTrainingLoss(y, 'bad')
        with self.assertRaises(ValueError):
            GNNTrainingLoss(y, 'combined', -1)


if __name__ == '__main__':
    unittest.main()
