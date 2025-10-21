# lgcp_daily_pyro.py
# -----------------------------------------------
# Log-Gaussian Cox Process for daily counts (Poisson point process aggregated by day)
# Inference via Pyro SVI (AutoDiagonalNormal guide).
# -----------------------------------------------

# pip install torch pyro-ppl pandas numpy matplotlib

import os
import math
import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam

# -----------------------
# Utilities
# -----------------------

def read_daily_csv(path, sep=";"):
    """
    Reads a CSV with columns: date; all_vessel
    Returns a DataFrame with a complete daily index (no gaps), filling missing days with 0 counts.
    """
    df = pd.read_csv(path, sep=sep)
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    assert "date" in df.columns and "all_vessel" in df.columns, "CSV must have columns 'date' and 'all_vessel'."

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    # Fill missing days
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    df.index.name = "date"
    df["all_vessel"] = df["all_vessel"].fillna(0).astype(int)

    # Optional exposure column; default to 1 if absent
    if "exposure" in df.columns:
        df["exposure"] = df["exposure"].fillna(1.0).astype(float)
    else:
        df["exposure"] = 1.0

    # Day index (0..N-1)
    df["t_idx"] = np.arange(len(df))
    return df.reset_index()


def normalize_time_index(t_idx):
    """
    Normalize integer day index to [0, 1] for kernel stability.
    """
    t = torch.tensor(t_idx, dtype=torch.get_default_dtype())
    t0 = t.min()
    t1 = t.max()
    if t1.item() == t0.item():
        return torch.zeros_like(t)  # degenerate single point case
    return (t - t0) / (t1 - t0 + 1e-9)


def matern32_kernel(x, y, variance, lengthscale):
    """
    Matérn-3/2 kernel K(x,y) = σ^2 * (1 + sqrt(3) r/ℓ) * exp(-sqrt(3) r/ℓ)
    where r = |x - y|.
    x: [N], y: [M]
    variance: scalar > 0
    lengthscale: scalar > 0
    returns: [N, M]
    """
    # compute pairwise distances
    x = x.unsqueeze(-1)               # [N,1]
    y = y.unsqueeze(0)                # [1,M]
    r = (x - y).abs()                 # [N,M]
    sqrt3 = math.sqrt(3.0)
    z = sqrt3 * r / lengthscale
    K = variance * (1.0 + z) * torch.exp(-z)
    return K


# -----------------------
# LGCP Model
# -----------------------

class LGCPDaily:
    """
    Log-Gaussian Cox Process for daily counts:
      y_d ~ Poisson( exposure_d * exp(alpha + f_d) )
      f ~ GP(0, K_Matern32)

    Priors:
      alpha ~ Normal(0, 5)
      log_lengthscale ~ Normal(0, 1)    (so ℓ ~ LogNormal(0,1))
      log_var ~ Normal(0, 1)            (so σ_f^2 ~ LogNormal(0,1))
    """

    def __init__(self, t01, y, exposure=None, jitter=1e-5, device="cpu"):
        """
        t01: torch.tensor [N] in [0,1]
        y: torch.tensor [N] nonnegative integers
        exposure: torch.tensor [N] positive (default ones)
        """
        self.device = device
        self.t01 = t01.to(device)
        self.y = y.to(device)
        N = len(y)
        if exposure is None:
            exposure = torch.ones(N, dtype=torch.get_default_dtype())
        self.exposure = exposure.to(device)
        self.jitter = torch.tensor(jitter, dtype=torch.get_default_dtype(), device=device)

        self.N = N

    def model(self):
        N = self.N
        t = self.t01

        # Hyperpriors
        alpha = pyro.sample("alpha", dist.Normal(0.0, 5.0))
        log_lengthscale = pyro.sample("log_lengthscale", dist.Normal(1.0, 0.3))
        log_var = pyro.sample("log_var", dist.Normal(0.0, 0.5))

        lengthscale = torch.exp(log_lengthscale) + 1e-6
        variance = torch.exp(log_var) + 1e-6

        # GP prior for latent f
        K = matern32_kernel(t, t, variance=variance, lengthscale=lengthscale)
        K = K + self.jitter * torch.eye(N, dtype=K.dtype, device=K.device)

        f = pyro.sample("f", dist.MultivariateNormal(loc=torch.zeros(N, device=K.device), covariance_matrix=K))

        # Intensity and likelihood
        rate = self.exposure * torch.exp(alpha + f)  # positive
        with pyro.plate("days", N):
            pyro.sample("y", dist.Poisson(rate), obs=self.y)

    def fit(self, num_steps=4000, lr=0.02, seed=123):
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)

        guide = AutoDiagonalNormal(self.model)
        optim = ClippedAdam({"lr": lr, "betas": (0.9, 0.999)})
        svi = SVI(self.model, guide, optim, loss=Trace_ELBO())

        losses = []
        for step in range(1, num_steps + 1):
            loss = svi.step()
            losses.append(loss)
            if step % 500 == 0:
                print(f"[step {step}] ELBO: {-loss:.1f}")

        self.guide = guide
        self.losses = losses
        return guide

    @torch.no_grad()
    def posterior_samples(self, num_samples=500):
        """
        Draw joint posterior samples of alpha, f, (hyperparameters).
        """
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=("alpha","f","log_lengthscale","log_var"))
        samples = predictive()
        # samples: dict mapping name -> [S, ...]
        return samples

    @torch.no_grad()
    def posterior_intensity_summary(self, num_samples=1000, exposure=None):
        """
        Returns dict with posterior summaries on the intensity per day:
          - rate_mean, rate_p05, rate_p50, rate_p95  (shape [N])
          - f_mean (latent log-intensity component)
        """
        if exposure is None:
            exposure = self.exposure
        S = num_samples
        samples = self.posterior_samples(num_samples=S)
        alpha_s = samples["alpha"]                    # [S]
        f_s = samples["f"]                            # [S, N]
        rate_s = exposure.unsqueeze(0) * torch.exp(alpha_s.unsqueeze(-1) + f_s)  # [S,N]

        def pct(q):
            return torch.quantile(rate_s, q, dim=0)

        out = {
            "rate_mean": rate_s.mean(dim=0),
            "rate_p05": pct(0.05),
            "rate_p50": pct(0.50),
            "rate_p95": pct(0.95),
            "f_mean": f_s.mean(dim=0),
        }
        return out


# -----------------------
# Script entry (example)
# -----------------------

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=False, default=None,
                        help="Path to CSV with columns: date;all_vessel[;exposure]. Default: uses a small synthetic example.")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)  # better numerical stability for GP

    if args.csv is None:
        # --- tiny example if you don't pass a CSV ---
        print("No --csv provided; running a small synthetic demo.")
        dates = pd.date_range("2023-01-02", periods=100, freq="D")
        # True latent intensity: seasonal-ish curve
        t = np.arange(len(dates))
        lam_true = np.exp(-1.0 + 0.8*np.sin(2*np.pi*t/30.0))
        rng = np.random.default_rng(0)
        y = rng.poisson(lam_true)
        demo = pd.DataFrame({"date": dates, "all_vessel": y})
        df = demo
    else:
        df = read_daily_csv(args.csv, sep=";")

    # Build tensors
    t_idx = df["t_idx"].values if "t_idx" in df.columns else np.arange(len(df))
    t01 = normalize_time_index(t_idx)

    y = torch.tensor(df["all_vessel"].values, dtype=torch.get_default_dtype())
    # Handle exposure safely
    if "exposure" in df.columns:
        exposure = torch.tensor(df["exposure"].values, dtype=torch.get_default_dtype())
    else:
        exposure = torch.ones(len(df), dtype=torch.get_default_dtype())
    # Fit model
    model = LGCPDaily(t01=t01, y=y, exposure=exposure, device="cpu")
    model.fit(num_steps=args.steps, lr=args.lr, seed=args.seed)

    # Posterior summaries
    summary = model.posterior_intensity_summary(num_samples=1000)
    rate_mean = summary["rate_mean"].cpu().numpy()
    rate_p05 = summary["rate_p05"].cpu().numpy()
    rate_p50 = summary["rate_p50"].cpu().numpy()
    rate_p95 = summary["rate_p95"].cpu().numpy()

    # Plot
    dates = pd.to_datetime(df["date"].values)
    plt.figure(figsize=(11, 5))
    plt.plot(dates, df["all_vessel"].values, label="Observed counts (y)")
    plt.plot(dates, rate_mean.squeeze(), label="Posterior mean rate")
    plt.fill_between(dates, rate_p05.squeeze(), rate_p95.squeeze(), alpha=0.25, label="90% posterior band")
    plt.title("LGCP: daily ships per day")
    #plot only 2 months of x-axis
    if len(dates) > 20:
        plt.xlim(dates[0], dates[19])
    plt.xlabel("Date")
    plt.ylabel("Expected ships/day")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("lgcp_daily_pyro_output.png")

    samples = model.posterior_samples(num_samples=1000)

    # alpha: global intercept
    alpha_s = samples["alpha"].numpy()
    print(f"Alpha (global log-intensity): mean={alpha_s.mean():.3f}, std={alpha_s.std():.3f}")

    # GP hyperparameters
    log_lengthscale_s = samples["log_lengthscale"].numpy()
    lengthscale_s = np.exp(log_lengthscale_s)
    print(f"Lengthscale: mean={lengthscale_s.mean():.3f}, std={lengthscale_s.std():.3f}")

    log_var_s = samples["log_var"].numpy()
    variance_s = np.exp(log_var_s)
    print(f"GP variance: mean={variance_s.mean():.3f}, std={variance_s.std():.3f}")


    # Print a small text summary
    #print("\nPosterior daily intensity (first 10 days):")
    #for i in range(min(10, len(df))):
        #d = str(pd.to_datetime(df['date'].iloc[i]).date())
        #print(f"{d}: mean={rate_mean.squeeze()[i]:.2f}, p50={rate_p50.squeeze()[i]:.2f}, [p05={rate_p05.squeeze()[i]:.2f}, p95={rate_p95.squeeze()[i]:.2f}]")
