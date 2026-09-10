"""
ConvLSTM baseline for Poisson count prediction.

Adapted from CATCH (Convolutional-LSTM approach for temporal catch
hotspots): Contreras Lopez et al., "Convolutional-LSTM approach for
temporal catch hotspots (CATCH): an AI-driven model for spatiotemporal
forecasting of fisheries catch probability densities",
https://pmc.ncbi.nlm.nih.gov/articles/PMC12203189/

Adaptations for this project's setting
---------------------------------------
CATCH forecasts *monthly* fish-catch *probability densities* on a large
regular ocean-basin raster, from CPUE + 4 static environmental covariates
(bottom temperature, depth, dissolved oxygen, salinity), trained with a
weighted binary cross-entropy loss and a 9x9 convolution kernel. Our
setting differs on every one of those axes: *daily* AIS vessel *counts*
(not normalized densities) on a small ~14x7 bounding grid (only 49 of the
98 cells fall inside the Gulf of Trieste), using the same covariates as
every other baseline in this project (none of CATCH's CPUE/depth/DO/
salinity are available here). We keep CATCH's core architectural idea --
stacked ConvLSTM layers processing a sequence of lagged spatial frames,
followed by a convolutional output head -- but:

  - predict a Poisson rate (softplus output head, Poisson NLL loss)
    instead of a normalized probability density, so results are directly
    comparable to every other baseline in this project on the same
    metrics;
  - use this project's fixed covariate set (chl, thetao, fishing_block,
    is_holiday, is_weekend) instead of CATCH's CPUE/depth/DO/salinity;
  - use a 3x3 (not 9x9) convolution kernel, since the grid here is far
    smaller than CATCH's basin-scale raster;
  - concatenate the target day's own (known-in-advance) covariate frame
    to the final hidden state before the output head, since chl/thetao/
    calendar covariates are genuinely available same-day here (unlike
    CATCH's forecasting setting) and every other baseline in this project
    is given the same information.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Grid indexing
# ---------------------------------------------------------------------------


def build_cell_grid_index(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Map irregular grid cells to integer (row, col) positions on the
    smallest enclosing regular lon/lat raster (same index-recovery logic
    as build_grid_adjacency in src.models.gnn, but returning positions
    instead of an adjacency matrix).

    Parameters
    ----------
    coords : [N, 2] array of (longitude, latitude), one row per grid cell.

    Returns
    -------
    row_idx, col_idx : [N] integer arrays (row = latitude axis, col = longitude axis)
    grid_shape : (n_rows, n_cols)
    """
    coords = np.asarray(coords, dtype=np.float64)
    lon, lat = coords[:, 0], coords[:, 1]

    uniq_lon = np.sort(np.unique(lon))
    uniq_lat = np.sort(np.unique(lat))

    lon_steps = np.diff(uniq_lon)
    lat_steps = np.diff(uniq_lat)
    lon_step = float(np.min(lon_steps)) if len(lon_steps) else 1.0
    lat_step = float(np.min(lat_steps)) if len(lat_steps) else 1.0

    col_idx = np.rint((lon - uniq_lon.min()) / lon_step).astype(int)
    row_idx = np.rint((lat - uniq_lat.min()) / lat_step).astype(int)

    grid_shape = (int(row_idx.max()) + 1, int(col_idx.max()) + 1)
    return row_idx, col_idx, grid_shape


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ConvLSTMCell(nn.Module):
    """
    Standard ConvLSTM cell (Shi et al., 2015, "Convolutional LSTM Network"):
    input-to-state and state-to-state transforms are both convolutions
    instead of the fully-connected operations used by a plain LSTM cell.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        # One conv producing all four gates (input, forget, output, cell) at once.
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        # x: [B, C_in, H, W]   h, c: [B, C_hidden, H, W]
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device, dtype):
        h_dim, w_dim = spatial_size
        h = torch.zeros(batch_size, self.hidden_channels, h_dim, w_dim, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_channels, h_dim, w_dim, device=device, dtype=dtype)
        return h, c


class ConvLSTMBaseline(nn.Module):
    """
    Stack of ConvLSTM layers processing a sequence of T lagged spatial
    frames (covariates + target of each past day), followed by a
    convolutional Poisson-rate output head. The target day's own static
    covariate frame is concatenated to the final hidden state before the
    head (see module docstring for why).

    forward(x_seq, x_static, mask) -> rate [B, H, W], strictly positive.
    """

    def __init__(
        self,
        seq_channels: int,
        static_channels: int,
        hidden_channels: int = 4,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    seq_channels if layer == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size,
                )
                for layer in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Conv2d(hidden_channels + static_channels, 1, kernel_size=3, padding=1)

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x_seq: [B, T, C_seq, H, W]   x_static: [B, C_static, H, W]
        B, T, _, H, W = x_seq.shape
        device, dtype = x_seq.device, x_seq.dtype

        layer_input = x_seq
        for cell in self.cells:
            h, c = cell.init_hidden(B, (H, W), device, dtype)
            outputs = []
            for t in range(T):
                h, c = cell(layer_input[:, t], h, c)
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)  # [B, T, hidden, H, W]
            layer_input = self.dropout(layer_input)

        final_h = layer_input[:, -1]  # [B, hidden, H, W]
        combined = torch.cat([final_h, x_static], dim=1)
        rate = nn.functional.softplus(self.head(combined)).squeeze(1)  # [B, H, W]

        if mask is not None:
            rate = rate * mask

        return rate


def poisson_nll_grid(rate: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean Poisson negative log-likelihood over valid (masked) grid cells
    only, up to an additive constant (log(y!) dropped, as elsewhere in
    this project).
    """
    nll = rate - y * torch.log(rate + eps)
    nll = nll * mask
    return nll.sum() / mask.sum().clamp(min=1.0)
