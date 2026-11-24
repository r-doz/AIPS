"""GNN and ConvLSTM training for the reviewer-1 experiments, with a Poisson or a hurdle head.

The training loops replicate scripts/15_train_gnn.py and scripts/18_train_convlstm.py (same data
pipeline, split, standardization, initialization order, optimizer, full-batch epochs, seeds and
determinism settings); only the output head and the loss can change:

  head = "poisson": rate = softplus(.), Poisson NLL                 (the definitive baselines)
  head = "hurdle" : logit p and rate lambda = softplus(.), hurdle NLL
                    -[b log p + (1-b) log(1-p) + b (y log lam - lam - log(1 - e^-lam))],  b = 1[y > 0]

A ConvLSTM config may list cell_lag_1 / cell_lag_7 in covariate_cols, together with a lag_features
block: the lags are then built exactly as for Red-LGCP and the GNN (observed past only) and enter
every covariate frame, so the target-day frame carries the counts of t-1 and t-7.

Each task trains one model on all days before a week and predicts that week; it returns one row per
cell-day with the rate (Poisson) or p and lam (hurdle).
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from r1_common import ROOT, cell_key, load_module

_gnn = load_module("r1_gnn", "scripts/15_train_gnn.py")
_conv = load_module("r1_conv", "scripts/18_train_convlstm.py")
from src.models.convlstm import ConvLSTMBaseline, build_cell_grid_index, poisson_nll_grid  # noqa: E402
from src.models.data_pp_lgcp import add_lag_features, make_day_split_masks  # noqa: E402
from src.models.gnn import SpatioTemporalGNN, build_grid_adjacency, normalize_adjacency, poisson_nll  # noqa: E402


# ---------------------------------------------------------------- hurdle heads and loss
def hurdle_nll(logit, rate, y, mask=None):
    b = (y > 0).float()
    lam = rate.clamp_min(1e-6)
    ztp = lam + torch.log(-torch.expm1(-lam)) - y * torch.log(lam)
    nll = F.binary_cross_entropy_with_logits(logit, b, reduction="none") + b * ztp
    if mask is None:
        return nll.mean()
    nll = nll * mask
    return nll.sum() / (mask.sum() * nll.shape[0]).clamp(min=1.0)


class HurdleGNN(SpatioTemporalGNN):
    """SpatioTemporalGNN trunk with a two-output head (activity logit, rate)."""

    def __init__(self, in_dim, hidden_dim=32, num_layers=2, dropout=0.1):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, adj_norm):
        h = x
        for layer in self.layers:
            h = self.dropout(self.activation(layer(h, adj_norm)))
        out = self.head(h)
        return out[..., 0], F.softplus(out[..., 1])


class HurdleConvLSTM(ConvLSTMBaseline):
    """ConvLSTMBaseline trunk with a two-channel convolutional head (activity logit, rate)."""

    def __init__(self, seq_channels, static_channels, hidden_channels=4, num_layers=2, kernel_size=3, dropout=0.2):
        super().__init__(seq_channels, static_channels, hidden_channels, num_layers, kernel_size, dropout)
        self.head = nn.Conv2d(hidden_channels + static_channels, 2, kernel_size=3, padding=1)

    def forward(self, x_seq, x_static, mask=None):
        B, T, _, H, W = x_seq.shape
        layer_input = x_seq
        for cell in self.cells:
            h, c = cell.init_hidden(B, (H, W), x_seq.device, x_seq.dtype)
            outputs = []
            for t in range(T):
                h, c = cell(layer_input[:, t], h, c)
                outputs.append(h)
            layer_input = self.dropout(torch.stack(outputs, dim=1))
        out = self.head(torch.cat([layer_input[:, -1], x_static], dim=1))
        return out[:, 0], F.softplus(out[:, 1])


# ---------------------------------------------------------------- data (cached per worker)
_CACHE: dict = {}


def _gnn_data(cfg):
    key = ("gnn", tuple(cfg["covariate_cols"]), str(cfg.get("lag_features")))
    if key not in _CACHE:
        df = _gnn.load_parquet_years(str(ROOT / cfg["parquet_path"]), cfg.get("years"))
        df["date"] = pd.to_datetime(df["date"])
        df_feat, feature_cols = _gnn.build_node_features(df, cfg)
        X, Y, dates, cell_coords = _gnn.build_graph_tensors(df_feat, feature_cols, cfg["target_col"])
        adj = normalize_adjacency(build_grid_adjacency(cell_coords[["longitude", "latitude"]].to_numpy(),
                                                       connectivity=cfg.get("adjacency", "queen")))
        _CACHE[key] = (X, Y, dates, cell_coords, torch.tensor(adj, dtype=torch.float32))
    return _CACHE[key]


def _conv_data(cfg):
    seq_length = int(cfg["seq_length"])
    key = ("conv", tuple(cfg["covariate_cols"]), str(cfg.get("lag_features")), seq_length)
    if key not in _CACHE:
        df = _conv.load_parquet_years(str(ROOT / cfg["parquet_path"]), cfg.get("years"))
        df["date"] = pd.to_datetime(df["date"])
        if (cfg.get("lag_features") or {}).get("enabled", False):
            df, _ = add_lag_features(df, cfg["lag_features"])
        df, covs = _conv.prepare_dataframe(df, cfg)
        cov_grid, y_grid, dates, mask, cell_coords, _ = _conv.build_grid_tensors(df, covs, cfg["target_col"])
        X_seq, X_static, Y, tdates = _conv.build_sequence_examples(cov_grid, y_grid, dates, seq_length)
        _CACHE[key] = (X_seq, X_static, Y, tdates, mask, cell_coords)
    return _CACHE[key]


def _rows(dates, cell_coords, y, out, head):
    """Long table for one week: dates [D], y [D, N], out = rate [D, N] or (p, lam)."""
    n = len(cell_coords)
    ci, cj = cell_key(cell_coords["longitude"], cell_coords["latitude"])
    res = pd.DataFrame({"date": pd.to_datetime(np.repeat(dates, n)), "ci": np.tile(ci, len(dates)),
                        "cj": np.tile(cj, len(dates)), "y": y.reshape(-1)})
    if head == "poisson":
        res["rate"] = out.reshape(-1)
    else:
        res["p"], res["lam"] = out[0].reshape(-1), out[1].reshape(-1)
    return res


# ---------------------------------------------------------------- one model, one week
def train_predict(task: dict) -> pd.DataFrame:
    cfg, head, seed = task["cfg"], task["head"], int(task["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    split = dict(train_fraction=cfg.get("train_fraction", 0.9), random_seed=cfg.get("data_seed", 42),
                 split_strategy="fixed_test_window", test_start_date=task["start"], test_end_date=task["end"])

    if task["kind"] == "gnn":
        X, Y, dates, cell_coords, adj = _gnn_data(cfg)
        tr, te = make_day_split_masks(df=pd.DataFrame({"date": dates}), **split)
        nf = X.shape[2]
        sc = StandardScaler().fit(X[tr].reshape(-1, nf))
        Xtr = torch.tensor(sc.transform(X[tr].reshape(-1, nf)).reshape(X[tr].shape), dtype=torch.float32)
        Xte = torch.tensor(sc.transform(X[te].reshape(-1, nf)).reshape(X[te].shape), dtype=torch.float32)
        Ytr = torch.tensor(Y[tr], dtype=torch.float32)
        Model = SpatioTemporalGNN if head == "poisson" else HurdleGNN
        model = Model(in_dim=nf, hidden_dim=int(cfg["hidden_dim"]), num_layers=int(cfg["num_layers"]),
                      dropout=float(cfg["dropout"]))
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        model.train()
        for _ in range(int(cfg["num_epochs"])):
            opt.zero_grad()
            out = model(Xtr, adj)
            loss = poisson_nll(out, Ytr) if head == "poisson" else hurdle_nll(*out, Ytr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xte, adj)
        if head == "poisson":
            out = np.clip(out.numpy(), 0, None)
        else:
            out = (torch.sigmoid(out[0]).numpy(), out[1].numpy())
        res = _rows(dates[te], cell_coords, Y[te], out, head)

    else:  # convlstm
        X_seq, X_static, Y, tdates, mask, cell_coords = _conv_data(cfg)
        tr, te = make_day_split_masks(df=pd.DataFrame({"date": tdates}), **split)
        sm, ss = _conv.fit_channel_scaler(X_seq[tr], 1)
        tm, ts = _conv.fit_channel_scaler(X_static[tr], 1)
        t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
        Xs_tr, Xs_te = t((X_seq[tr] - sm) / ss), t((X_seq[te] - sm) / ss)
        Xt_tr, Xt_te = t((X_static[tr] - tm) / ts), t((X_static[te] - tm) / ts)
        Ytr, mask_t = t(Y[tr]), t(mask)
        Model = ConvLSTMBaseline if head == "poisson" else HurdleConvLSTM
        model = Model(seq_channels=Xs_tr.shape[2], static_channels=Xt_tr.shape[1],
                      hidden_channels=int(cfg["hidden_channels"]), num_layers=int(cfg["num_layers"]),
                      kernel_size=int(cfg["kernel_size"]), dropout=float(cfg["dropout"]))
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
        model.train()
        for _ in range(int(cfg["num_epochs"])):
            opt.zero_grad()
            out = model(Xs_tr, Xt_tr, mask_t)
            loss = poisson_nll_grid(out, Ytr, mask_t) if head == "poisson" else hurdle_nll(*out, Ytr, mask_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xs_te, Xt_te, mask_t)
        ri, cj, _ = build_cell_grid_index(cell_coords[["longitude", "latitude"]].to_numpy())
        if head == "poisson":
            out = np.clip(out.numpy(), 0, None)[:, ri, cj]
        else:
            out = (torch.sigmoid(out[0]).numpy()[:, ri, cj], out[1].numpy()[:, ri, cj])
        res = _rows(tdates[te], cell_coords, Y[te][:, ri, cj], out, head)

    for k in ("model", "seed", "split", "week", "month"):
        res.insert(0, k, task[k])
    for k, v in (task.get("tags") or {}).items():
        res[k] = v
    return res


def run_tasks(tasks: list[dict], workers: int | None = None) -> pd.DataFrame:
    """Run tasks in parallel processes (one thread each), returning their rows in task order."""
    workers = workers or int(os.environ.get("R1_WORKERS", 6))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(train_predict, tasks, chunksize=1))
    return pd.concat(parts, ignore_index=True)


def make_tasks(model: str, kind: str, head: str, cfg: dict, weeks, seeds, tags: dict | None = None) -> list[dict]:
    return [dict(model=model, kind=kind, head=head, cfg=copy.deepcopy(cfg), seed=s, split=split, week=name,
                 start=start, end=end, month=month, tags=tags)
            for (split, name, start, end, month) in weeks for s in seeds]


def load_cfg(kind: str, path: str, overrides: dict | None = None) -> dict:
    mod = _gnn if kind == "gnn" else _conv
    cfg = mod.load_config(str(ROOT / path))
    for k, v in (overrides or {}).items():
        cfg[k] = v
    return cfg
