"""
Joint hurdle variant of the sparse variational LGCP.

A neural classifier and the LGCP are trained together in a single ELBO with a
hurdle likelihood:

    P(y = 0)          = 1 - p
    P(y = k), k >= 1  = p * Poisson(k | lambda) / (1 - exp(-lambda))

where p = sigmoid(g(x)) is the classifier's probability that the cell-day is
active and lambda = exp(alpha + beta^T z + f) is the LGCP intensity, so the
LGCP describes the counts of active cell-days (zero-truncated Poisson). The
classifier has dropout, and Monte Carlo dropout at prediction time gives a
per-cell-day uncertainty that can modulate the gate threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lgcp import SparseLGCP, build_Kfu, build_Kuu, kl_qp


def _log1mexp_neg(eta: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(-lam)) for lam = exp(eta) > 0, stable for tiny and large lam."""
    exact = torch.log(-torch.expm1(-lam.clamp_min(1e-12)))
    return torch.where(eta < -10.0, eta, exact)


def _conditional_mean(lam: torch.Tensor) -> torch.Tensor:
    """E[y | y > 0] = lam / (1 - exp(-lam)) for a Poisson(lam) count."""
    exact = lam / (-torch.expm1(-lam.clamp_min(1e-12)))
    return torch.where(lam < 1e-6, 1.0 + 0.5 * lam, exact)


class HurdleSparseLGCP(SparseLGCP):
    def __init__(
        self,
        coords_np,
        covariates_np,
        y_np,
        kernel_config: dict,
        M_inducing: int = 300,
        device: str = "cpu",
        np_seed: int = 0,
        clf_hidden: int = 32,
        clf_dropout: float = 0.1,
        clf_alpha: float = 1e-4,
    ):
        super().__init__(
            coords_np, covariates_np, y_np, kernel_config,
            M_inducing=M_inducing, device=device, np_seed=np_seed,
        )
        in_dim = self.X.shape[1] + self.covs.shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, clf_hidden),
            nn.ReLU(),
            nn.Dropout(clf_dropout),
            nn.Linear(clf_hidden, 1),
        ).to(self.device)
        active_rate = float((self.y > 0).float().mean().clamp(1e-4, 1 - 1e-4))
        with torch.no_grad():
            self.classifier[-1].bias.fill_(float(np.log(active_rate / (1.0 - active_rate))))
        self.clf_alpha = float(clf_alpha)

    def _clf_features(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        return torch.cat([X, C], dim=1)

    def elbo_mc(self, num_mc: int = 4, minibatch_size: int | None = None, jitter: float = 1e-6):
        kern = self.kernel_params()
        Z = self.Z
        eye_M = torch.eye(self.M, device=self.device, dtype=Z.dtype)
        Kuu = build_Kuu(Z, kern, jitter)
        cholKuu = torch.linalg.cholesky(Kuu + 1e-8 * eye_M)
        S = self.get_S()
        kl = kl_qp(self.m, S, Kuu, jitter=jitter)

        if minibatch_size is None or minibatch_size >= self.N:
            idx = torch.arange(self.N, device=self.device)
            n_batch = self.N
        else:
            idx = torch.randperm(self.N, device=self.device)[:minibatch_size]
            n_batch = minibatch_size
        X_b, cov_b, y_b = self.X[idx], self.covs[idx], self.y[idx]
        pos = (y_b > 0).to(y_b.dtype)

        logit = self.classifier(self._clf_features(X_b, cov_b)).squeeze(-1)
        ll_clf = (pos * F.logsigmoid(logit) + (1.0 - pos) * F.logsigmoid(-logit)).sum()

        Kfu = build_Kfu(X_b, Z, kern)
        A = torch.cholesky_solve(Kfu.T, cholKuu).T
        Lq = torch.linalg.cholesky(S + 1e-8 * eye_M)
        eps = torch.randn(num_mc, self.M, device=self.device, dtype=Z.dtype)
        U = self.m.unsqueeze(0) + eps @ Lq.T
        eta = self.alpha + (cov_b @ self.beta).unsqueeze(0) + U @ A.T
        lam = torch.exp(eta)
        log_trunc = y_b * eta - lam - torch.lgamma(y_b + 1.0) - _log1mexp_neg(eta, lam)
        ll_cnt = (log_trunc * pos).sum(dim=1).mean()

        penalty = 0.5 * self.clf_alpha * sum(
            (layer.weight ** 2).sum() for layer in self.classifier if isinstance(layer, nn.Linear)
        )
        scale = float(self.N) / float(n_batch)
        elbo = scale * (ll_clf + ll_cnt) - kl - penalty
        return elbo, {
            "elbo_ll": float((ll_clf + ll_cnt).item()),
            "ll_clf": float(ll_clf.item()),
            "ll_cnt": float(ll_cnt.item()),
            "kl": float(kl.item()),
            "scale": scale,
        }

    @torch.no_grad()
    def predict_hurdle(self, coords_new, covs_new, num_samples: int = 300, num_dropout: int = 100, jitter: float = 1e-6):
        """Return (p_mean, p_std, cond_mean, lam_mean) for each new cell-day.

        p_mean / p_std: Monte Carlo dropout mean and std of P(active);
        cond_mean: E_q[lambda / (1 - exp(-lambda))], the expected count given activity;
        lam_mean: E_q[lambda].
        """
        dtype = torch.get_default_dtype()
        Xnew = torch.tensor(coords_new, dtype=dtype, device=self.device)
        Cnew = torch.tensor(covs_new, dtype=dtype, device=self.device)

        kern = self.kernel_params()
        Z = self.Z
        eye_M = torch.eye(self.M, device=self.device, dtype=dtype)
        Kuu = build_Kuu(Z, kern, jitter)
        cholKuu = torch.linalg.cholesky(Kuu + 1e-8 * eye_M)
        A_new = torch.cholesky_solve(build_Kfu(Xnew, Z, kern).T, cholKuu).T
        Lq = torch.linalg.cholesky(self.get_S() + 1e-8 * eye_M)
        eps = torch.randn(num_samples, self.M, device=self.device, dtype=dtype)
        U = self.m.unsqueeze(0) + eps @ Lq.T
        eta = self.alpha + (Cnew @ self.beta).unsqueeze(0) + U @ A_new.T
        lam = torch.exp(eta)
        cond_mean = _conditional_mean(lam).mean(0)
        lam_mean = lam.mean(0)

        was_training = self.classifier.training
        self.classifier.train()
        feats = self._clf_features(Xnew, Cnew)
        probs = torch.stack(
            [torch.sigmoid(self.classifier(feats).squeeze(-1)) for _ in range(num_dropout)]
        )
        self.classifier.train(was_training)
        return (
            probs.mean(0).cpu().numpy(),
            probs.std(0).cpu().numpy(),
            cond_mean.cpu().numpy(),
            lam_mean.cpu().numpy(),
        )


def apply_modulated_gate(mu, p, tau, dates, gamma: float = 1.0, scale: float = 1.0) -> np.ndarray:
    """Confidence-weighted redistribution with a per-cell-day threshold tau.

    Cells with p < tau are zeroed; each returns the fraction scale * p / tau of its
    expected count, which is shared among the accepted cells of the same day in
    proportion to mu * p**gamma.
    """
    mu = np.asarray(mu, dtype=float)
    p = np.asarray(p, dtype=float)
    tau = np.broadcast_to(np.asarray(tau, dtype=float), mu.shape)
    accepted = p >= tau
    out = np.where(accepted, mu, 0.0)
    days = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    for day in days.unique():
        mask = np.asarray(days == day)
        acc, rej = mask & accepted, mask & ~accepted
        if not acc.any():
            continue
        recovered = np.sum(mu[rej] * scale * p[rej] / tau[rej])
        if recovered == 0:
            continue
        weights = mu[acc] * np.power(p[acc], gamma)
        if weights.sum() == 0:
            weights = np.power(p[acc], gamma)
        out[acc] += recovered * weights / weights.sum()
    return out
