import unittest
import numpy as np
from src.visualization.grid_surface import complete_grid_surface


class GridSurfaceTests(unittest.TestCase):
    def test_virtual_cells_and_diffusion_conservation(self):
        axis = np.arange(4.)
        x, y = np.meshgrid(axis, axis)
        keep = ~((x == 1) & (y == 1))
        values = np.zeros(15)
        values[0] = 9
        raw = complete_grid_surface(x[keep], y[keep], values, axis, axis,
                                    diffusion_strength=0)
        self.assertEqual(raw[1, 1], 0)
        for strength in (0.25, 1):
            spread = complete_grid_surface(x[keep], y[keep], values, axis, axis,
                                           diffusion_strength=strength)
            self.assertAlmostEqual(spread.sum(), 9)
            self.assertGreater(spread[1, 1], 0)

    def test_smooth_surface_bounded_and_no_remote_hotspot(self):
        axis = np.arange(5.)
        x, y = np.meshgrid(axis, axis)
        values = np.zeros_like(x)
        values[2, 1] = 4
        dense = np.linspace(0, 4, 100)
        result = complete_grid_surface(x.ravel(), y.ravel(), values.ravel(), dense, dense,
                                       diffusion_strength=0)
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 4)
        np.testing.assert_allclose(result[:, dense >= 3], 0)
