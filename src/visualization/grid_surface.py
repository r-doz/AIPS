"""Display surfaces from a zero-filled rectangular grid (no RBF fitting)."""
import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator


def complete_grid_surface(x, y, values, xi, yi, radius_cells=1,
                          diffusion_strength=0.25, display_method='pchip'):
    """Diffuse on the full rectangle, then interpolate for display only.

    Missing cells start at zero. Edge-normalized diffusion conserves the
    rectangular grid total; subsequent interpolation/masking is not a count grid.
    """
    if not isinstance(radius_cells, int) or radius_cells < 0:
        raise ValueError('radius_cells must be a nonnegative integer')
    if not np.isfinite(diffusion_strength) or not 0 <= diffusion_strength <= 1:
        raise ValueError('diffusion_strength must be between 0 and 1')
    if display_method not in ('pchip', 'linear'):
        raise ValueError('display_method must be pchip or linear')

    def axis(v):
        unique = np.unique(v)
        if len(unique) < 2:
            raise ValueError('At least two grid coordinates per axis are required')
        step = np.min(np.diff(unique))
        indices = np.rint((unique - unique[0]) / step).astype(int)
        if not np.allclose(unique, unique[0] + indices * step, atol=step * .002, rtol=0):
            raise ValueError('Spatial coordinates must form a regular grid')
        return unique[0] + np.arange(indices.max() + 1) * step, step

    x, y, values = (np.asarray(a, dtype=float) for a in (x, y, values))
    if x.ndim != 1 or x.shape != y.shape or x.shape != values.shape:
        raise ValueError('Coordinates and values must be aligned one-dimensional arrays')
    if not all(np.isfinite(a).all() for a in (x, y, values)) or (values < 0).any():
        raise ValueError('Requires finite coordinates and nonnegative finite values')
    gx, dx = axis(x)
    gy, dy = axis(y)
    ix = np.rint((x - gx[0]) / dx).astype(int)
    iy = np.rint((y - gy[0]) / dy).astype(int)
    if len(np.unique(np.c_[ix, iy], axis=0)) != len(values):
        raise ValueError('Duplicate spatial cells')
    full = np.zeros((len(gy), len(gx)))
    full[iy, ix] = values
    kernel = np.ones((2 * radius_cells + 1,) * 2)
    destinations = convolve(np.ones_like(full), kernel, mode='constant', cval=0)
    spread = convolve(full / destinations, kernel, mode='constant', cval=0)
    spread = (1 - diffusion_strength) * full + diffusion_strength * spread
    qx = np.clip(xi, gx[0], gx[-1])
    qy = np.clip(yi, gy[0], gy[-1])
    if display_method == 'pchip' and min(len(gx), len(gy)) >= 4:
        along_x = PchipInterpolator(gx, spread, axis=1)(qx)
        return PchipInterpolator(gy, along_x, axis=0)(qy)
    X, Y = np.meshgrid(qx, qy)
    return RegularGridInterpolator((gy, gx), spread)(np.c_[Y.ravel(), X.ravel()]).reshape(X.shape)
