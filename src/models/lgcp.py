"""
src/lgcp.py
Sparse Variational Log-Gaussian Cox Process (LGCP) for spatio-temporal vessel counts.

Architecture
------------
  coords  : [N, 3]  (lon_std, lat_std, t_std)  — all standardized on TRAIN set only
  covs    : [N, 2]  (chl_std, thetao_std)       — standardized on TRAIN set only
  y       : [N]     non-negative integer counts

Intensity model:
  log λ(s,t) = α + β·x + f(s,t)
  f(s,t) ~ GP(0, k_s(s,s') · k_t(t,t'))   (separable space-time kernel)
  k_s : RBF,  k_t : Periodic

Inference:
  Sparse variational GP:  q(u) = N(m, L Lᵀ)  with M inducing points.
  ELBO = E_q [ log p(y|f) ] − KL(q(u) ∥ p(u))
  Poisson log-likelihood:  log p(y|λ) = y log λ − λ − log y!
  Optimized with Adam + gradient clipping.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


def rbf_kernel(
    x: torch.Tensor, y: torch.Tensor, lengthscale: torch.Tensor, variance: torch.Tensor
) -> torch.Tensor:
    """
    Squared-exponential (RBF) kernel.
    x : [N, D],  y : [M, D]  →  [N, M]
    k(x,y) = σ² exp(−‖x−y‖² / (2 ℓ²))
    """
    # Numerically stable distance via (x-y)²  expansion
    x2 = (x**2).sum(dim=1, keepdim=True)  # [N, 1]
    y2 = (y**2).sum(dim=1, keepdim=True)  # [M, 1]
    d2 = x2 + y2.T - 2.0 * (x @ y.T)
    d2 = d2.clamp(min=0.0)  # avoid tiny negatives from float rounding
    return variance * torch.exp(-0.5 * d2 / lengthscale**2)


def periodic_kernel_1d(
    t1: torch.Tensor,
    t2: torch.Tensor,
    period: torch.Tensor,
    lengthscale: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """
    Standard periodic (Mackay) kernel.
    t1 : [N, 1],  t2 : [M, 1]  →  [N, M]
    k(t,t') = σ² exp(−2 sin²(π|t−t'|/p) / ℓ²)
    """
    diff = (t1 - t2.T).abs()
    arg = torch.sin(math.pi * diff / period)
    return variance * torch.exp(-2.0 * arg**2 / lengthscale**2)


# ---------------------------------------------------------------------------
# Kernel matrix builders
# ---------------------------------------------------------------------------


def _kern_matrices(
    X: torch.Tensor, Z: torch.Tensor, kern: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Ks [Nx, M], Kt [Nx, M]) for data points X and inducing Z."""
    Ks = rbf_kernel(X[:, :2], Z[:, :2], kern["spatial_length"], kern["spatial_var"])
    Kt = periodic_kernel_1d(
        X[:, 2:3], Z[:, 2:3], kern["time_period"], kern["time_length"], kern["time_var"]
    )
    return Ks, Kt


def build_Kuu(Z: torch.Tensor, kern: dict, jitter: float = 1e-6) -> torch.Tensor:
    """
    Prior covariance at inducing points: K_uu = K_s ⊙ K_t   [M, M].
    Separable space-time kernel (Hadamard product = product of kernels
    applied to decoupled inputs).
    """
    Ks, Kt = _kern_matrices(Z, Z, kern)
    Kuu = Ks * Kt
    noise = kern.get("noise_var", 0.0)
    Kuu = Kuu + (noise + jitter) * torch.eye(Z.shape[0], device=Z.device, dtype=Z.dtype)
    return Kuu


def build_Kfu(X: torch.Tensor, Z: torch.Tensor, kern: dict) -> torch.Tensor:
    """Cross-covariance between data and inducing points: K_fu [N, M]."""
    Ks, Kt = _kern_matrices(X, Z, kern)
    return Ks * Kt


# ---------------------------------------------------------------------------
# KL divergence  KL( q(u) = N(m, S) ∥ p(u) = N(0, K) )
# ---------------------------------------------------------------------------


def kl_qp(
    m: torch.Tensor, S: torch.Tensor, K: torch.Tensor, jitter: float = 1e-6
) -> torch.Tensor:
    """
    KL( N(m, S) ∥ N(0, K) ) = ½ [ tr(K⁻¹ S) + mᵀ K⁻¹ m − M + log|K| − log|S| ]
    Uses Cholesky solves for numerical stability.
    """
    M = m.shape[0]
    eye_M = torch.eye(M, device=K.device, dtype=K.dtype)

    cholK = torch.linalg.cholesky(K + jitter * eye_M)
    # K⁻¹ m
    Kinv_m = torch.cholesky_solve(m.unsqueeze(-1), cholK).squeeze(-1)
    # tr(K⁻¹ S)
    trace_term = torch.trace(torch.cholesky_solve(S, cholK))
    # log|K| = 2 Σ log diag(cholK)
    logdetK = 2.0 * torch.log(torch.diagonal(cholK)).sum()

    cholS = torch.linalg.cholesky(S + jitter * eye_M)
    logdetS = 2.0 * torch.log(torch.diagonal(cholS)).sum()

    mKm = (m * Kinv_m).sum()
    return 0.5 * (trace_term + mKm - M + logdetK - logdetS)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SparseLGCP(nn.Module):
    """
    Sparse variational Log-Gaussian Cox Process.

    Parameters
    ----------
    coords_np   : [N, 3]   (lon_std, lat_std, t_std)
    covariates_np : [N, C]  standardized covariates
    y_np        : [N]      integer counts
    M_inducing  : int      number of inducing points
    device      : str
    np_seed     : int      for reproducible inducing-point initialization
    """

    def __init__(
        self,
        coords_np,
        covariates_np,
        y_np,
        M_inducing: int = 300,
        device: str = "cpu",
        np_seed: int = 0,
    ):
        super().__init__()
        self.device = torch.device(device)
        dtype = torch.get_default_dtype()

        self.X = torch.tensor(coords_np, dtype=dtype, device=self.device)
        self.covs = torch.tensor(covariates_np, dtype=dtype, device=self.device)
        self.y = torch.tensor(y_np, dtype=dtype, device=self.device)
        self.N = self.X.shape[0]
        self.C = self.covs.shape[1]

        # ---- Kernel hyperparameters (log-parameterized for positivity) --------
        self.log_spatial_length = nn.Parameter(
            torch.tensor(math.log(0.5), device=self.device)
        )
        self.log_spatial_var = nn.Parameter(
            torch.tensor(math.log(1.0), device=self.device)
        )
        self.log_time_length = nn.Parameter(
            torch.tensor(math.log(0.3), device=self.device)
        )
        self.log_time_var = nn.Parameter(
            torch.tensor(math.log(0.5), device=self.device)
        )
        # Period fixed to 1.0 in standardized time — learnable so the model can
        # discover sub-annual periodicity if present.
        self.log_time_period = nn.Parameter(
            torch.tensor(math.log(1.0), device=self.device)
        )
        self.log_noise = nn.Parameter(torch.tensor(math.log(1e-3), device=self.device))

        # ---- Linear predictor --------------------------------------------------
        self.alpha = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.beta = nn.Parameter(torch.zeros(self.C, device=self.device))

        # ---- Inducing points ---------------------------------------------------
        self.M = M_inducing
        rng = np.random.default_rng(np_seed)
        idx = rng.choice(self.N, size=self.M, replace=False)
        Z_init = torch.tensor(coords_np[idx], dtype=dtype, device=self.device)
        self.Z = nn.Parameter(Z_init)  # [M, 3]  — jointly optimized

        # ---- Variational parameters  q(u) = N(m, L Lᵀ) -----------------------
        self.m = nn.Parameter(torch.zeros(self.M, device=self.device))

        # Parameterize L as lower-triangular.
        # Store the raw (unconstrained) lower triangle; diagonal is passed through
        # softplus to guarantee S ≻ 0 without adding a separate jitter term.
        self._L_raw = nn.Parameter(1e-3 * torch.eye(self.M, device=self.device))

    # ------------------------------------------------------------------
    def _get_L(self) -> torch.Tensor:
        """Return lower-triangular Cholesky factor L with softplus diagonal."""
        L = torch.tril(self._L_raw)
        # Replace diagonal with softplus to ensure strict positivity
        diag_pos = F.softplus(torch.diagonal(self._L_raw)) + 1e-6
        L = L - torch.diag(torch.diagonal(L)) + torch.diag(diag_pos)
        return L

    def get_S(self) -> torch.Tensor:
        """Return S = L Lᵀ."""
        L = self._get_L()
        return L @ L.T

    def kernel_params(self) -> dict:
        return {
            "spatial_length": torch.exp(self.log_spatial_length),
            "spatial_var": torch.exp(self.log_spatial_var),
            "time_length": torch.exp(self.log_time_length),
            "time_var": torch.exp(self.log_time_var),
            "time_period": torch.exp(self.log_time_period),
            "noise_var": torch.exp(self.log_noise),
        }

    # ------------------------------------------------------------------
    def elbo_mc(
        self, num_mc: int = 4, minibatch_size: int | None = None, jitter: float = 1e-6
    ) -> tuple[torch.Tensor, dict]:
        """
        Monte Carlo ELBO:
          ELBO = (N/B) · E_q [ Σ_b log p(y_b | λ_b) ] − KL(q(u) ∥ p(u))

        The Poisson log-likelihood includes log y! (via lgamma) so that the
        reported value is a proper log-likelihood, not just the sufficient
        statistic.  This does not affect gradients wrt model parameters.
        """
        kern = self.kernel_params()
        Z = self.Z

        # ---- Prior covariance at inducing points ----
        Kuu = build_Kuu(Z, kern, jitter)
        eye_M = torch.eye(self.M, device=self.device, dtype=Z.dtype)
        cholKuu = torch.linalg.cholesky(Kuu + 1e-8 * eye_M)

        # ---- KL ----
        S = self.get_S()
        kl = kl_qp(self.m, S, Kuu, jitter=jitter)

        # ---- Minibatch selection ----
        if minibatch_size is None or minibatch_size >= self.N:
            idx = torch.arange(self.N, device=self.device)
            n_batch = self.N
        else:
            idx = torch.randperm(self.N, device=self.device)[:minibatch_size]
            n_batch = minibatch_size

        X_b = self.X[idx]  # [B, 3]
        cov_b = self.covs[idx]  # [B, C]
        y_b = self.y[idx]  # [B]

        # ---- Predictive mean weights:  A = K_fu K_uu^{-1}  [B, M] ----
        Kfu = build_Kfu(X_b, Z, kern)  # [B, M]
        A = torch.cholesky_solve(Kfu.T, cholKuu).T  # [B, M]

        # ---- MC samples from q(u) ----
        Lq = torch.linalg.cholesky(S + 1e-8 * eye_M)
        eps = 1e-9
        ll_sum = torch.zeros(1, device=self.device)

        for _ in range(num_mc):
            z = torch.randn(self.M, device=self.device)
            u = self.m + Lq @ z  # [M]
            f = A @ u  # [B]
            lin = self.alpha + (cov_b @ self.beta) + f
            lam = torch.exp(lin)
            # Full Poisson log-likelihood (includes − log y! for proper LL)
            log_y_fact = torch.lgamma(y_b + 1.0)
            logp = (y_b * torch.log(lam + eps) - lam - log_y_fact).sum()
            ll_sum = ll_sum + logp

        elbo_ll = ll_sum / num_mc
        scale = float(self.N) / float(n_batch)
        elbo = scale * elbo_ll - kl

        return elbo, {
            "elbo_ll": elbo_ll.item(),
            "kl": kl.item(),
            "scale": scale,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_rate(
        self,
        coords_new: np.ndarray,
        covs_new: np.ndarray,
        num_samples: int = 200,
        jitter: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo predictive distribution of the Poisson rate λ.

        Returns
        -------
        rate_mean : [N_new]   posterior mean of λ
        rate_p05  : [N_new]   5th percentile
        rate_p95  : [N_new]   95th percentile
        """
        dtype = torch.get_default_dtype()
        Xnew = torch.tensor(coords_new, dtype=dtype, device=self.device)
        Cnew = torch.tensor(covs_new, dtype=dtype, device=self.device)

        kern = self.kernel_params()
        Z = self.Z
        eye_M = torch.eye(self.M, device=self.device, dtype=dtype)
        Kuu = build_Kuu(Z, kern, jitter)
        cholKuu = torch.linalg.cholesky(Kuu + 1e-8 * eye_M)

        Kfu_new = build_Kfu(Xnew, Z, kern)
        # A_new = K_fu_new K_uu^{-1}  [N_new, M]
        A_new = torch.cholesky_solve(Kfu_new.T, cholKuu).T

        S = self.get_S()
        Lq = torch.linalg.cholesky(S + 1e-8 * eye_M)

        rates = []
        for _ in range(num_samples):
            z = torch.randn(self.M, device=self.device)
            u = self.m + Lq @ z
            f = A_new @ u
            lin = self.alpha + (Cnew @ self.beta) + f
            rates.append(torch.exp(lin))

        rates = torch.stack(rates, dim=0)  # [S, N_new]
        rate_mean = rates.mean(0).cpu().numpy()
        rate_p05 = torch.quantile(rates, 0.05, dim=0).cpu().numpy()
        rate_p95 = torch.quantile(rates, 0.95, dim=0).cpu().numpy()
        return rate_mean, rate_p05, rate_p95
