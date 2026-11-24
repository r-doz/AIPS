"""Red-LGCP trained in two separate stages (rebuttal experiment on the two-stage ablation).

The joint ELBO of Red-LGCP (src/models/hurdle_lgcp.py) is

    N/B [ sum_i log Bern(1{y_i > 0} | p_i)  +  sum_{i: y_i > 0} log ZTP(y_i | lambda_i) ]  -  KL(q(u) || p(u))  -  L2(classifier)

and the classifier (p) and the count model (alpha, beta, kernel, Z, q(u)) share no parameters, so each
part only receives gradients from its own terms. HurdleTwoStage trains the two parts one after the other,
each with a fresh optimizer and the joint run's settings (Adam, lr, minibatch, num_mc, num_steps, gradient
clipping):
  stage "clf": the MC-dropout classifier alone (Bernoulli log-likelihood + L2 penalty);
  stage "cnt": the zero-truncated Poisson LGCP alone (positive counts) + KL.
Compared with joint training, the gradient-norm clipping acts on each part separately and the two parts no
longer share minibatches or random streams; the model and the objective are the same.

Prediction, gate (tau + sigma, rho = gamma = 1) and outputs are those of scripts/11c_train_hurdle_lgcp.py,
which runs unchanged with the model class and the training function swapped.

Usage: python twostage_hurdle.py --config <per-run config of a definitive Red-LGCP run>
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

from src.models.hurdle_lgcp import HurdleSparseLGCP, _log1mexp_neg  # noqa: E402
from src.models.lgcp import build_Kfu, build_Kuu, kl_qp  # noqa: E402

STAGES = ("clf", "cnt")


class HurdleTwoStage(HurdleSparseLGCP):
    """HurdleSparseLGCP whose ELBO can be restricted to the classifier or to the count model."""

    stage = "joint"

    def elbo_mc(self, num_mc: int = 4, minibatch_size: int | None = None, jitter: float = 1e-6):
        if self.stage == "joint":
            return super().elbo_mc(num_mc=num_mc, minibatch_size=minibatch_size, jitter=jitter)
        if minibatch_size is None or minibatch_size >= self.N:
            idx = torch.arange(self.N, device=self.device)
            n_batch = self.N
        else:
            idx = torch.randperm(self.N, device=self.device)[:minibatch_size]
            n_batch = minibatch_size
        X_b, cov_b, y_b = self.X[idx], self.covs[idx], self.y[idx]
        pos = (y_b > 0).to(y_b.dtype)
        scale = float(self.N) / float(n_batch)

        if self.stage == "clf":
            logit = self.classifier(self._clf_features(X_b, cov_b)).squeeze(-1)
            ll_clf = (pos * F.logsigmoid(logit) + (1.0 - pos) * F.logsigmoid(-logit)).sum()
            penalty = 0.5 * self.clf_alpha * sum(
                (layer.weight ** 2).sum() for layer in self.classifier if isinstance(layer, nn.Linear))
            return scale * ll_clf - penalty, {"elbo_ll": float(ll_clf.item()), "ll_clf": float(ll_clf.item()),
                                              "ll_cnt": 0.0, "kl": 0.0, "scale": scale}

        assert self.stage == "cnt", self.stage
        kern = self.kernel_params()
        Z = self.Z
        eye_M = torch.eye(self.M, device=self.device, dtype=Z.dtype)
        Kuu = build_Kuu(Z, kern, jitter)
        cholKuu = torch.linalg.cholesky(Kuu + 1e-8 * eye_M)
        S = self.get_S()
        kl = kl_qp(self.m, S, Kuu, jitter=jitter)
        Kfu = build_Kfu(X_b, Z, kern)
        A = torch.cholesky_solve(Kfu.T, cholKuu).T
        Lq = torch.linalg.cholesky(S + 1e-8 * eye_M)
        U = self.m.unsqueeze(0) + torch.randn(num_mc, self.M, device=self.device, dtype=Z.dtype) @ Lq.T
        eta = self.alpha + (cov_b @ self.beta).unsqueeze(0) + U @ A.T
        lam = torch.exp(eta)
        log_trunc = y_b * eta - lam - torch.lgamma(y_b + 1.0) - _log1mexp_neg(eta, lam)
        ll_cnt = (log_trunc * pos).sum(dim=1).mean()
        return scale * ll_cnt - kl, {"elbo_ll": float(ll_cnt.item()), "ll_clf": 0.0, "ll_cnt": float(ll_cnt.item()),
                                     "kl": float(kl.item()), "scale": scale}


def load_train_module():
    spec = importlib.util.spec_from_file_location("hurdle_train", ROOT / "scripts/11c_train_hurdle_lgcp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    mod = load_train_module()
    joint_train = mod.base.train

    def two_stage_train(model, cfg):
        losses = []
        for stage in STAGES:
            model.stage = stage
            print(f"=== stage {stage}: {cfg['num_steps']} steps, fresh Adam ===", flush=True)
            losses += joint_train(model, cfg)
        model.stage = "joint"
        return losses

    mod.HurdleSparseLGCP = HurdleTwoStage
    mod.base.train = two_stage_train
    mod.main(mod.base.load_config(args.config))


if __name__ == "__main__":
    main()
