"""Red-LGCP with the diagonal SVGP residual variance restored (rebuttal experiment).

Red-LGCP uses the subset-of-regressors projection f_R = A_R u, so q(f_R) = N(A_R m, A_R S A_R^T)
(Sec. 4.2). The standard SVGP marginal (Hensman et al., 2015) adds the diagonal residual

    r_i = k(x_i, x_i) - K_iu K_uu^{-1} K_ui = k(x_i, x_i) - sum_j A_ij K_ij,

giving f_i ~ N(a_i^T m, a_i^T S a_i + r_i). The hurdle likelihood factorizes over cell-days, so only
these marginals enter the ELBO and the predictive distribution: f_i = a_i^T u + sqrt(r_i) eps_i with
independent eps_i ~ N(0, 1). All kernels of the model are stationary, so k(x_i, x_i) is one constant
k0; the extra cost is O(BM) per step (a row sum of A * K_fu, both already computed).

HurdleSVGP overrides elbo_mc and predict_hurdle of HurdleSparseLGCP and nothing else. Training reuses
scripts/11c_train_hurdle_lgcp.py unchanged, with the model class swapped.

Usage: python svgp_residual.py --config <per-run config> [--variant svgp|sor]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.hurdle_lgcp import HurdleSparseLGCP, _conditional_mean, _log1mexp_neg  # noqa: E402
from src.models.lgcp import _canonical_kernel_name, _combine_spacetime, _kern_matrices, build_Kfu, build_Kuu, kl_qp  # noqa: E402

STATIONARY = {"rbf", "periodic", "quasi_periodic", "rational_quadratic"}


def prior_variance(kern: dict, x0: torch.Tensor) -> torch.Tensor:
    """k(x, x) of the space-time kernel (without the inducing-point nugget); constant for stationary kernels."""
    for group in ("spatial", "temporal"):
        for comp in kern[group]:
            assert _canonical_kernel_name(comp["type"]) in STATIONARY, comp["type"]
    Ks, Kt = _kern_matrices(x0[:1], x0[:1], kern)
    return _combine_spacetime(Ks, Kt, kern)[0, 0]


def residual_variance(kern: dict, X: torch.Tensor, Kfu: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """diag(K_RR - A_R K_uR) for the rows of X, clamped at 0."""
    return (prior_variance(kern, X) - (A * Kfu).sum(dim=1)).clamp_min(0.0)


class HurdleSVGP(HurdleSparseLGCP):
    """HurdleSparseLGCP whose latent marginals include the SVGP residual variance."""

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
        r = residual_variance(kern, X_b, Kfu, A)                                    # new
        eps_f = torch.randn(num_mc, n_batch, device=self.device, dtype=Z.dtype)      # new
        eta = (self.alpha + (cov_b @ self.beta).unsqueeze(0) + U @ A.T
               + torch.sqrt(r.clamp_min(1e-12)).unsqueeze(0) * eps_f)                # new term
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
            "mean_residual_var": float(r.mean().item()),
        }

    @torch.no_grad()
    def predict_hurdle(self, coords_new, covs_new, num_samples: int = 300, num_dropout: int = 100, jitter: float = 1e-6):
        dtype = torch.get_default_dtype()
        Xnew = torch.tensor(coords_new, dtype=dtype, device=self.device)
        Cnew = torch.tensor(covs_new, dtype=dtype, device=self.device)

        kern = self.kernel_params()
        Z = self.Z
        eye_M = torch.eye(self.M, device=self.device, dtype=dtype)
        Kuu = build_Kuu(Z, kern, jitter)
        cholKuu = torch.linalg.cholesky(Kuu + 1e-8 * eye_M)
        Kfu = build_Kfu(Xnew, Z, kern)
        A_new = torch.cholesky_solve(Kfu.T, cholKuu).T
        Lq = torch.linalg.cholesky(self.get_S() + 1e-8 * eye_M)
        eps = torch.randn(num_samples, self.M, device=self.device, dtype=dtype)
        U = self.m.unsqueeze(0) + eps @ Lq.T
        r = residual_variance(kern, Xnew, Kfu, A_new)                                       # new
        eps_f = torch.randn(num_samples, Xnew.shape[0], device=self.device, dtype=dtype)     # new
        eta = self.alpha + (Cnew @ self.beta).unsqueeze(0) + U @ A_new.T + torch.sqrt(r).unsqueeze(0) * eps_f
        lam = torch.exp(eta)
        cond_mean = _conditional_mean(lam).mean(0)
        lam_mean = lam.mean(0)

        was_training = self.classifier.training
        self.classifier.train()
        feats = self._clf_features(Xnew, Cnew)
        probs = torch.stack([torch.sigmoid(self.classifier(feats).squeeze(-1)) for _ in range(num_dropout)])
        self.classifier.train(was_training)
        return (probs.mean(0).cpu().numpy(), probs.std(0).cpu().numpy(),
                cond_mean.cpu().numpy(), lam_mean.cpu().numpy())


def load_train_module():
    spec = importlib.util.spec_from_file_location("svgp_11c", ROOT / "scripts/11c_train_hurdle_lgcp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", choices=["svgp", "sor"], default="svgp")
    args = ap.parse_args()
    mod = load_train_module()
    if args.variant == "svgp":
        mod.HurdleSparseLGCP = HurdleSVGP   # 11c's main() builds the model through this name
    mod.main(mod.base.load_config(args.config))


if __name__ == "__main__":
    main()
