"""Calendar-day periods must retain their meaning across training cutoffs."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lgcp_trainer', ROOT / 'scripts/11_train_lgcp.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class PeriodDaysTests(unittest.TestCase):
    def test_calendar_conversion_and_model_construction(self):
        for i in range(1, 9):
            cfg = trainer.load_config(ROOT / f'config/exp{i}.yaml')
            original = copy.deepcopy(cfg['kernel_config'])
            cutoff = np.datetime64(cfg['test_start_date'])
            days = np.arange(int((cutoff - np.datetime64('2024-01-01')).astype(int)))
            # Match prepare_data's two-year normalization and float32 input.
            span = 730
            scaler = StandardScaler().fit((days / span).astype(np.float32).reshape(-1, 1))
            resolved = trainer.resolve_period_days(original, {'t_scaler': scaler}, span)
            hp = resolved['temporal'][1]['hyperparameters']
            week = scaler.transform(np.array([[0.0], [7 / span]]))
            self.assertAlmostEqual(hp['period'], float(week[1, 0] - week[0, 0]))
            self.assertEqual(cfg['kernel_config'], original)
            self.assertNotIn('period_days', hp)
            model = trainer.SparseLGCP(
                np.zeros((4, 3), dtype=np.float32),
                np.zeros((4, 7), dtype=np.float32), np.ones(4),
                kernel_config=resolved, M_inducing=2, device='cpu', np_seed=0,
            )
            params = model.kernel_params()['temporal'][1]['params']
            self.assertAlmostEqual(float(params['period']), hp['period'], places=6)
            self.assertFalse(params['period'].requires_grad)

    def test_legacy_period_unchanged(self):
        cfg = trainer.load_config(ROOT / 'config/train_basic_lgcp_multi_kernel.yaml')
        scaler = StandardScaler().fit([[0.0], [1.0]])
        self.assertEqual(trainer.resolve_period_days(cfg['kernel_config'], {'t_scaler': scaler}, 730), cfg['kernel_config'])

    def test_invalid_periods(self):
        cfg = trainer.load_config(ROOT / 'config/exp1.yaml')
        block = cfg['model']['temporal_kernels'][1]
        for value in [0, -1, float('nan'), float('inf')]:
            invalid = copy.deepcopy(block)
            invalid['hyperparameters']['period_days'] = value
            with self.assertRaises(ValueError):
                trainer._normalize_kernel_block(invalid, 'model.temporal_kernels[1]')
        invalid = copy.deepcopy(block)
        invalid['hyperparameters']['period'] = 0.074
        with self.assertRaises(ValueError):
            trainer._normalize_kernel_block(invalid, 'model.temporal_kernels[1]')
        with self.assertRaises(ValueError):
            trainer._normalize_kernel_block(block, 'model.spatial_kernels[1]')


if __name__ == '__main__':
    unittest.main()
