"""
Spatial Graph Neural Network baseline for Poisson count prediction.

Nodes are the fixed 0.5' grid cells of the Gulf of Trieste; edges connect
spatially adjacent cells. For each date, node features (covariates + lag
features) are propagated through a small stack of GCN layers (Kipf & Welling,
2017) to predict a per-cell Poisson rate.

This is a plain-torch implementation (no torch_geometric dependency): the
grid has only ~50 nodes, so a dense adjacency matrix is used throughout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_grid_adjacency(
    coords: np.ndarray,
    connectivity: str = "queen",
) -> np.ndarray:
    """
    Build a dense adjacency matrix for a (possibly irregular) regular grid.

    Parameters
    ----------
    coords : [N, 2] array of (longitude, latitude), one row per grid cell.
    connectivity : "rook" (4-neighbours, shared edge) or
                   "queen" (8-neighbours, shared edge or corner).

    Returns
    -------
    adj : [N, N] binary adjacency matrix (no self-loops).
    """
    if connectivity not in ("rook", "queen"):
        raise ValueError(f"Unknown connectivity: {connectivity}")

    coords = np.asarray(coords, dtype=np.float64)
    lon, lat = coords[:, 0], coords[:, 1]

    lon_steps = np.diff(np.sort(np.unique(lon)))
    lat_steps = np.diff(np.sort(np.unique(lat)))

    lon_step = float(np.min(lon_steps)) if len(lon_steps) else 1.0
    lat_step = float(np.min(lat_steps)) if len(lat_steps) else 1.0

    # Grid indices (integer column/row) for each cell.
    col_idx = np.rint((lon - lon.min()) / lon_step).astype(int)
    row_idx = np.rint((lat - lat.min()) / lat_step).astype(int)

    n = len(coords)
    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        d_col = np.abs(col_idx - col_idx[i])
        d_row = np.abs(row_idx - row_idx[i])

        if connectivity == "rook":
            neighbour_mask = ((d_col == 1) & (d_row == 0)) | (
                (d_col == 0) & (d_row == 1)
            )
        else:  # queen
            neighbour_mask = (
                (d_col <= 1) & (d_row <= 1) & ~((d_col == 0) & (d_row == 0))
            )

        adj[i, neighbour_mask] = 1.0

    # Symmetrize defensively (should already be symmetric by construction).
    adj = np.maximum(adj, adj.T)

    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Symmetric GCN normalization: D^-1/2 (A + I) D^-1/2.
    """
    n = adj.shape[0]
    adj_self = adj + np.eye(n, dtype=adj.dtype)
    degree = adj_self.sum(axis=1)
    deg_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    deg_inv_sqrt[degree == 0] = 0.0

    return (deg_inv_sqrt[:, None] * adj_self) * deg_inv_sqrt[None, :]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GCNLayer(nn.Module):
    """
    One graph-convolution layer: H' = A_norm @ (H @ W + b).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x : [batch, n_nodes, in_dim]
        # adj_norm : [n_nodes, n_nodes]
        support = self.linear(x)
        return torch.einsum("ij,bjf->bif", adj_norm, support)


class SpatioTemporalGNN(nn.Module):
    """
    Stack of GCN layers followed by a per-node Poisson rate head.

    forward(x, adj_norm) -> rate [batch, n_nodes], strictly positive.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.activation(layer(h, adj_norm))
            h = self.dropout(h)

        rate = nn.functional.softplus(self.head(h)).squeeze(-1)
        return rate


def poisson_nll(rate: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean Poisson negative log-likelihood, up to an additive constant
    (the log(y!) term is dropped since it does not depend on the model).
    """
    return (rate - y * torch.log(rate + eps)).mean()


class GNNTrainingLoss(nn.Module):
    """Select cell Poisson, combined, or daily-only MAE training.

    Scales are fixed from training targets: max(mean cell count, 1) and
    max(mean daily total, 1). The cell_poisson mode is exactly the historical
    unnormalized objective. Daily-only supervision does not identify spatial
    allocations; use the summed forecast for downstream LGCP allocation.
    """

    def __init__(self, training_targets, mode='cell_poisson', daily_weight=1.0):
        super().__init__()
        if mode not in ('cell_poisson', 'combined', 'daily_only'):
            raise ValueError('loss.mode must be cell_poisson, combined, or daily_only')
        if not np.isfinite(daily_weight) or daily_weight < 0:
            raise ValueError('loss.daily_weight must be finite and nonnegative')
        y = training_targets.detach()
        if y.ndim != 2 or y.numel() == 0 or not torch.isfinite(y).all() or (y < 0).any():
            raise ValueError('Training targets must be finite nonnegative [days, cells] counts')
        self.mode = mode
        self.daily_weight = float(daily_weight)
        self.register_buffer('cell_scale', y.mean().clamp_min(1.0))
        self.register_buffer('daily_scale', y.sum(dim=1).mean().clamp_min(1.0))

    def forward(self, rate, y):
        if rate.ndim != 2 or rate.shape != y.shape:
            raise ValueError('Rates and targets must have the same [days, cells] shape')
        if self.mode == 'cell_poisson':
            return poisson_nll(rate, y)
        daily_mae = (rate.sum(dim=1) - y.sum(dim=1)).abs().mean() / self.daily_scale
        if self.mode == 'daily_only':
            return daily_mae
        return poisson_nll(rate, y) / self.cell_scale + self.daily_weight * daily_mae

    def metadata(self):
        return dict(mode=self.mode, daily_weight=self.daily_weight,
                    cell_scale=float(self.cell_scale.item()), daily_scale=float(self.daily_scale.item()))
