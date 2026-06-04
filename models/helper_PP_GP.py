# lgcp_aids_ais.py
# Sparse variational Log-Gaussian Cox Process for ais_vessels_count
# - coordinates: latitude, longitude, date -> (s_lat, s_lon, t_norm)
# - linear covariates: chl, thetao
# - latent f(s,t) approximated with inducing points + q(u)=N(m,S)
# - minibatch Monte Carlo ELBO optimization (ADAM)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.linalg as tla
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.path import Path
from scipy import ndimage
from scipy.interpolate import Rbf
from scipy.special import gammaln  # for log-factorial

# reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_default_dtype(torch.float32)

# -------------------------
# Training helper
# -------------------------
def train_model(coords, covs, y, M=300, lr=1e-2, num_steps=1000, minibatch=1024, device="cpu", num_mc=7, np_seed=0):
    model = SparseLGCP(coords, covs, y, M_inducing=M, device=device, np_seed=np_seed)
    minibatch = min(minibatch, coords.shape[0])
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(1, num_steps + 1):
        optim.zero_grad()
        elbo, info = model.elbo_mc(num_mc=num_mc, minibatch_size=minibatch)
        loss = -elbo
        loss.backward()
        # gradient clipping to help stabilize
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optim.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == 1:
           #print(f"[{step}/{num_steps}] - loss={loss.item():.3f} (ll_est={info['elbo_ll']:.3f} kl={info['kl']:.3f} scale={info['scale']:.1f})")
            print(f"{loss.item():.4f}")
    return model, losses


# ------------------------- Utilities 
def is_leap_year(year: int) -> bool:
    """
    Returns True if 'year' is a leap year, False otherwise.
    """
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


# -------------------------
# Data preparation function (from your parquet)
# -------------------------
def prepare_data_from_parquet(parquet_path, train_fraction=0.9, random_seed=42):
    """
    Loads parquet with columns:
    date (YYYY-MM-DD), latitude, longitude, chl, thetao, ais_vessels_count
    Definitions:
        - coords: key of each ds record ([longitude, latitude, t_norm])
        - covariates: predictor variables but not target
        - target: objective of the prediction  
    Returns:
      train_coords, train_covs, train_y
      test_coords, test_covs, test_y
      scalers: dict of fitted scalers (for later prediction)
      df: original DataFrame
    """

    df = pd.read_parquet(parquet_path)
    
    # Build the scalers  
    coord_scaler = StandardScaler() # x - mean / std 
    cov_scaler = StandardScaler()
    t_scaler = StandardScaler()

    # Build standardized coordinates
    df['date'] = pd.to_datetime(df['date'])
    year = df['date'].dt.year.iloc[0]
    days_in_year = 366.0 if is_leap_year(year) else 365.0
    df['t_norm'] = df['date'].dt.dayofyear.astype(float) / days_in_year
    t_norm = df['t_norm'].values.astype(np.float32).reshape(-1, 1)
    t_std = t_scaler.fit_transform(t_norm)

    coords_raw = df[['longitude', 'latitude']].values.astype(np.float32)
    coords_std = coord_scaler.fit_transform(coords_raw)

    coords = np.hstack([coords_std, t_std])  # [N,3]

    # Define standardized covariates and target
    covs_raw = df[['chl', 'thetao']].values.astype(np.float32)
    covs = cov_scaler.fit_transform(covs_raw)

    y = df['ais_vessels_count'].fillna(0).astype(np.int32).values

    # -------------------------------
    # Split into train/test by full days
    # -------------------------------
    unique_dates = np.sort(df["date"].unique())
    rng = np.random.default_rng(random_seed)

    num_train_days = int(train_fraction * len(unique_dates))
    rng.shuffle(unique_dates)

    train_dates = unique_dates[:num_train_days]
    test_dates = unique_dates[num_train_days:]

    train_mask = df["date"].isin(train_dates)
    test_mask = df["date"].isin(test_dates)

    train_coords = coords[train_mask]
    train_covs = covs[train_mask]
    train_y = y[train_mask]

    test_coords = coords[test_mask]
    test_covs = covs[test_mask]
    test_y = y[test_mask]
    scalers = {"coord_scaler": coord_scaler, "cov_scaler": cov_scaler, "t_scaler": t_scaler}
    
    return train_coords, train_covs, train_y, test_coords, test_covs, test_y, scalers, df


def compute_meta(df, grid_x=None, grid_y=None):
    """
    Compute metadata dictionary for spatio-temporal GP/PP models.

    Args:
        df: pandas DataFrame with columns ['date', 'latitude', 'longitude']
        grid_x, grid_y: optional (int) number of grid cells in x (lon) and y (lat)
    
    Returns:
        meta: dict with date/time and spatial normalization info
    """

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"])
    t_min, t_max = df["date"].min(), df["date"].max()

    # Time normalization (0, 1)
    dates_unique = np.sort(df["date"].unique())

    # Compute grid resolution if not given
    if grid_x is None:
        grid_x = len(np.unique(df["longitude"]))
    if grid_y is None:
        grid_y = len(np.unique(df["latitude"]))

    # Spatial normalization
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()

    meta = {
        "dates": dates_unique,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "t_min": t_min,
        "t_max": t_max,
    }

    return meta


def evaluate_test_metrics(y_test, rate_mean_test, dates_test):
    """
    Evaluate test metrics for Poisson predictions.
    
    Parameters
    ----------
    y_test : np.ndarray, shape [N_test]
        Observed counts.
    rate_mean_test : np.ndarray, shape [N_test]
        Predicted rates (lambda) from the model.
    dates_test : np.ndarray or pd.Series, shape [N_test]
        Corresponding dates for each observation (datetime or np.datetime64).
    
    Returns
    -------
    metrics : dict
        Dictionary containing:
            - mean_ll_obs : Mean log-likelihood per observation
            - rmse_daily : Daily RMSE
            - mean_ll_daily : Mean log-likelihood of daily totals
    """
    # Ensure dates are pd.Timestamp
    dates_test = pd.to_datetime(dates_test)
    
    # --- 1️⃣ Mean log-likelihood per observation ---
    # log(y!) = gammaln(y+1)
    log_fact = gammaln(y_test + 1)
    log_ll_obs = y_test * np.log(rate_mean_test + 1e-9) - rate_mean_test - log_fact
    mean_ll_obs = log_ll_obs.mean()
    
    # --- 2️⃣ Daily RMSE and 3️⃣ Daily mean log-likelihood ---
    df = pd.DataFrame({
        "date": dates_test,
        "y": y_test,
        "lambda_hat": rate_mean_test
    })
    
    daily = df.groupby("date").agg({
        "y": "sum",
        "lambda_hat": "sum"
    }).reset_index()
    
    # Daily RMSE
    rmse_daily = np.sqrt(np.mean((daily["y"] - daily["lambda_hat"])**2))
    
    # Daily mean log-likelihood
    log_fact_daily = gammaln(daily["y"].values + 1)
    ll_daily = daily["y"].values * np.log(daily["lambda_hat"].values + 1e-9) - daily["lambda_hat"].values - log_fact_daily
    mean_ll_daily = ll_daily.mean()
    
    metrics = {
        "mean_ll_obs": mean_ll_obs,
        "rmse_daily": rmse_daily,
        "mean_ll_daily": mean_ll_daily
    }
    
    return metrics


def plot_pp_overview(model, df, coords, covariate, y_true, meta, num_samples=200, target_date=None, test_coords=None, t_scaler = None, cmap="viridis"):
    """
    Plot total ships per day (pred vs. real) and spatial maps for one target date.
    
    Args:
        model: trained SparseLGCP model
        coords: (N, 3) tensor or np.array -> [lon, lat, t_norm]
        covariate: tensor or np.array of covariate(s)
        y_true: tensor or np.array of true counts
        meta: dict with 'dates', 'grid_x', 'grid_y', 't_min', 't_max'
        num_samples: Monte Carlo samples for rate prediction
        target_date: optional date string (e.g. "2024-07-15")
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    if torch.is_tensor(covariate):
        covariate = covariate.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    
    # Predict rate mean
    rate_mean, _, _ = model.predict_rate(coords, covariate, num_samples=num_samples)
    rate_mean = np.clip(rate_mean, 0, None)

    # -------------------------------
    # 1️⃣ Time Series: total ships per day
    # -------------------------------
    dates = meta["dates"]
    # Ensure consistent datetime64[ns] types
    dates = pd.to_datetime(meta["dates"])
    t_min = pd.to_datetime(meta["t_min"])
    t_max = pd.to_datetime(meta["t_max"])

    # Compute normalized time in [0, 1]
    t_norms = (dates - t_min) / (t_max - t_min)
    t_norms = t_norms.to_numpy(dtype=float)
    t_norms = np.array(t_norms, dtype=float)

    total_pred = []
    total_true = []

    for t in np.unique(coords[:, 2]):
        mask = np.isclose(coords[:, 2], t, atol=1e-4)
        total_pred.append(rate_mean[mask].sum())
        total_true.append(y_true[mask].sum())

    plt.figure(figsize=(10, 4))
    plt.plot(dates, total_true, label="True total ships", color="black", lw=2)
    plt.plot(dates, total_pred, label="Predicted total ships", color="tab:blue", lw=2)
    plt.xlabel("Date")
    plt.ylabel("Total ships")
    if test_coords is not None:
        # Invert standardization using t_scaler from prepare_data_from_parquet
        t_std_test = test_coords[:, 2].reshape(-1, 1)
        t_norm_test = t_scaler.inverse_transform(t_std_test).ravel()  # back to [0,1] scale

        # Convert normalized times back to actual dates
        t_min = pd.to_datetime(meta["t_min"])
        t_max = pd.to_datetime(meta["t_max"])

        test_dates = t_min + pd.to_timedelta(t_norm_test * (t_max - t_min))
        plt.scatter(test_dates, np.zeros(len(test_dates)),
            color="green", marker="|", s=200, label="Test dates")

    plt.title("Daily total ships (real vs predicted)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()




    

    # -------------------------------
    # 2️⃣ Select target date for spatial map
    # -------------------------------
    if target_date is None:
        target_date = dates[0]  # default: first date
    else:
        target_date = pd.Timestamp(target_date)

    t_norm = (target_date - meta["t_min"]) / (meta["t_max"] - meta["t_min"])
    # Find the closest time slice to the requested date
    time_diff = np.abs(coords[:, 2] - float(t_norm))
    closest_idx = np.argmin(time_diff)
    closest_t = coords[closest_idx, 2]

    mask = np.isclose(coords[:, 2], closest_t, atol=1e-5)
    closest_date = pd.to_datetime(meta["t_min"]) + pd.to_timedelta(
        closest_t * (meta["t_max"] - meta["t_min"])
    )

    print(f"Selected closest available date: {closest_date.date()} (normalized {closest_t:.3f})")
 



    # Infer grid bounds
    x, y = coords[mask, 0], coords[mask, 1]
    rm = rate_mean[mask]
    yt = y_true[mask]

    gulf_df = pd.read_csv('data/raw/ts_gulf_coords.csv')
    #gulf_x = (gulf_df["longitude"].to_numpy() - meta["lon_min"]) / (meta["lon_max"]- meta["lon_min"])
    #gulf_y = (gulf_df["latitude"].to_numpy() - meta["lat_min"]) / (meta["lat_max"] - meta["lat_min"])
    # Assuming coords are your GP input tensor
    x_mean, x_std = df["longitude"].mean(), df["longitude"].std()
    y_mean, y_std = df["latitude"].mean(), df["latitude"].std()


    gulf_x = (gulf_df["longitude"].to_numpy() - x_mean) / x_std
    gulf_y = (gulf_df["latitude"].to_numpy() - y_mean) / y_std


    # Define regular grid for interpolation (you can tune resolution)
    nx, ny = 150, 100
    #xi = np.linspace(x.min(), x.max(), nx)
    #yi = np.linspace(y.min(), y.max(), ny)
    xi = np.linspace(gulf_x.min(), gulf_x.max(), nx)
    yi = np.linspace(gulf_y.min(), gulf_y.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate the irregular data onto the grid
    #rm_grid = griddata((x, y), rm, (Xi, Yi), method="cubic") #cubic, linear, nearest
    #yt_grid = griddata((x, y), yt, (Xi, Yi), method="cubic")

    rbf_rm = Rbf(x, y, rm, function='gaussian')  # or 'multiquadric', 'gaussian', 'linear'
    rbf_yt = Rbf(x, y, yt, function='gaussian')

    rm_grid = rbf_rm(Xi, Yi)
    yt_grid = rbf_yt(Xi, Yi)

    # Ensure non-negative
    rm_grid = np.clip(rm_grid, 0, None)
    yt_grid = np.clip(yt_grid, 0, None)

    mask_nan = np.isnan(rm_grid)
    rm_grid[mask_nan] = ndimage.generic_filter(rm_grid, np.nanmean, size=3)[mask_nan]
    yt_grid[mask_nan] = ndimage.generic_filter(yt_grid, np.nanmean, size=3)[mask_nan]

    gulf_poly = Path(np.column_stack([gulf_x, gulf_y]))
    points = np.column_stack([Xi.flatten(), Yi.flatten()])
    inside_mask = gulf_poly.contains_points(points).reshape(Xi.shape)

    rm_masked = np.where(inside_mask, rm_grid, np.nan)
    yt_masked = np.where(inside_mask, yt_grid, np.nan)


    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    im0 = axs[0].imshow(rm_masked, origin="lower", extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                        cmap=cmap, aspect="auto")
    axs[0].plot(gulf_x, gulf_y, color="red", lw=1.2)
    axs[0].set_title(f"Predicted Intensity ({closest_date.date()})")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(yt_masked, origin="lower", extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                        cmap=cmap, aspect="auto")
    axs[1].plot(gulf_x, gulf_y, color="red", lw=1.2)
    axs[1].set_title(f"Observed Counts ({closest_date.date()})")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("Longitude (standardized)")
        ax.set_ylabel("Latitude (standardized)")

    plt.tight_layout()
    plt.show()

# -------------------------
# Kernel utilities
# -------------------------
def rbf_kernel(x, y, lengthscale, variance):
    # x: [N, D], y: [M, D] -> returns [N, M]
    x2 = (x**2).sum(dim=1, keepdim=True)    # [N,1]
    y2 = (y**2).sum(dim=1, keepdim=True)    # [M,1]
    d2 = x2 + y2.t() - 2.0 * (x @ y.t())
    l2 = lengthscale**2
    K = variance * torch.exp(-0.5 * d2 / l2)
    return K

def periodic_kernel_1d(t1, t2, period, lengthscale, variance):
    # t1: [N,1], t2: [M,1] -> [N,M]
    # k(t,t') = variance * exp( - 2 sin^2(pi|t-t'|/period) / ell^2 )
    diff = (t1 - t2.t()).abs()
    arg = torch.sin(math.pi * diff / period)
    K = variance * torch.exp(-2.0 * (arg**2) / (lengthscale**2))
    return K

# -------------------------
# Build Kuu and Kfu (space-time)
# -------------------------
#def build_Kuu(Z, kern_params, jitter):
#    # Z: [M,3] -> spatial (0:2) + time (2:3)
#    Z_space = Z[:, 0:2]   # lon, lat
#    Z_time = Z[:, 2:3]    # time normalized
#    K_s = rbf_kernel(Z_space, Z_space, kern_params["spatial_length"], kern_params["spatial_var"])
#    K_t = periodic_kernel_1d(Z_time, Z_time, kern_params["time_period"], kern_params["time_length"], kern_params["time_var"])
#    Kuu = K_s + K_t + (kern_params.get("noise_var", 0.0) + jitter) * torch.eye(Z.shape[0], device=Z.device)
#    return Kuu
#
#def build_Kfu(X, Z, kern_params):
#    # X: [B,3], Z: [M,3] -> returns [B, M]
#    X_space = X[:, 0:2]
#    Z_space = Z[:, 0:2]
#    X_time = X[:, 2:3]
#    Z_time = Z[:, 2:3]
#    Ks = rbf_kernel(X_space, Z_space, kern_params["spatial_length"], kern_params["spatial_var"])
#    Kt = periodic_kernel_1d(X_time, Z_time, kern_params["time_period"], kern_params["time_length"], kern_params["time_var"])
#    return Ks + Kt

def build_Kuu(Z, kern_params, jitter=1e-6):
    # Z: [M,3] -> spatial (0:2) + time (2:3)
    Z_space = Z[:, 0:2]   # spatial coords (e.g., lon, lat)
    Z_time  = Z[:, 2:3]   # time (normalized/standardized)

    K_s = rbf_kernel(
        Z_space, Z_space,
        kern_params["spatial_length"],
        kern_params["spatial_var"],
    )
    K_t = periodic_kernel_1d(
        Z_time, Z_time,
        kern_params["time_period"],
        kern_params["time_length"],
        kern_params["time_var"],
    )

    # Space-time separable kernel (captures interaction)
    K_st = K_s * K_t

    noise = kern_params.get("noise_var", 0.0)
    Kuu = K_st + (noise + jitter) * torch.eye(Z.shape[0], device=Z.device, dtype=Z.dtype)
    return Kuu


def build_Kfu(X, Z, kern_params):
    # X: [B,3], Z: [M,3] -> returns [B, M]
    X_space = X[:, 0:2]
    Z_space = Z[:, 0:2]
    X_time  = X[:, 2:3]
    Z_time  = Z[:, 2:3]

    K_s = rbf_kernel(
        X_space, Z_space,
        kern_params["spatial_length"],
        kern_params["spatial_var"],
    )
    K_t = periodic_kernel_1d(
        X_time, Z_time,
        kern_params["time_period"],
        kern_params["time_length"],
        kern_params["time_var"],
    )

    # Space-time separable kernel
    return K_s * K_t


# -------------------------
# KL between q(u)=N(m,S) and p(u)=N(0,K)
# -------------------------
def kl_qp(m, S, K, jitter=1e-6):
    # m: [M], S: [M,M], K: [M,M]
    M = m.shape[0]
    cholK = torch.linalg.cholesky(K + jitter * torch.eye(K.shape[0], device=K.device))
    # solve K^{-1} m
    Kinv_m = torch.cholesky_solve(m.unsqueeze(-1), cholK).squeeze(-1)
    trace_term = torch.trace(torch.cholesky_solve(S, cholK))
    logdetK = 2.0 * torch.log(torch.diagonal(cholK)).sum()
    cholS = torch.linalg.cholesky(S + 1e-8 * torch.eye(S.shape[0], device=S.device))
    logdetS = 2.0 * torch.log(torch.diagonal(cholS)).sum()
    mKm = (m * Kinv_m).sum()
    kl = 0.5 * (trace_term + mKm - M + logdetK - logdetS)
    return kl

# -------------------------
# Main model class
# -------------------------
class SparseLGCP(nn.Module):
    def __init__(self, coords_np, covariates_np, y_np, M_inducing=300, device="cpu", np_seed=0):
        """
        coords_np: [N,3] numpy array columns = [lon, lat, time_norm]
        covariates_np: [N,2] numpy columns = [chl, thetao] (standardized outside or will be)
        y_np: [N] counts (ints)
        """
        super().__init__()
        self.device = torch.device(device)
        self.X = torch.tensor(coords_np, dtype=torch.get_default_dtype(), device=self.device)      # [N,3]
        self.covs = torch.tensor(covariates_np, dtype=torch.get_default_dtype(), device=self.device) # [N,2]
        self.y = torch.tensor(y_np, dtype=torch.get_default_dtype(), device=self.device)           # [N]
        self.N = self.X.shape[0]

        # kernel hyperparameters (log-space for positivity)
        self.log_spatial_length = nn.Parameter(torch.tensor(math.log(0.5), device=self.device))
        self.log_spatial_var = nn.Parameter(torch.tensor(math.log(1.0), device=self.device))
        self.log_time_length = nn.Parameter(torch.tensor(math.log(0.2), device=self.device))
        self.log_time_var = nn.Parameter(torch.tensor(math.log(0.5), device=self.device))
        self.log_time_period = nn.Parameter(torch.tensor(math.log(1.0), device=self.device))  # period (in time-norm units)
        self.log_noise = nn.Parameter(torch.tensor(math.log(1e-3), device=self.device))

        # linear coefficients for covariates + intercept
        self.alpha = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.beta = nn.Parameter(torch.tensor([0.5, 0.5], device=self.device))  # [beta_chl, beta_thetao]

        # inducing points initialization
        #self.M = min(M_inducing, max(10, int(0.05 * self.N)))
        self.M = M_inducing
        # random subset of data coords for Z init
        rng = np.random.default_rng(np_seed)
        idx = rng.choice(self.N, size=self.M, replace=False)
        Z_init = torch.tensor(coords_np[idx, :], dtype=torch.get_default_dtype(), device=self.device)
        self.Z = nn.Parameter(Z_init)   # [M,3]

        # variational parameters q(u) = N(m, S = L L^T)
        self.m = nn.Parameter(torch.zeros(self.M, device=self.device))   # mean
        # parameterize L as lower triangular matrix stored unconstrained (we'll use tril)
        L_init = 1e-2 * torch.eye(self.M, device=self.device)
        self.L_unconstrained = nn.Parameter(L_init)

    def kernel_params(self):
        return {
            "spatial_length": torch.exp(self.log_spatial_length),
            "spatial_var": torch.exp(self.log_spatial_var),
            "time_length": torch.exp(self.log_time_length),
            "time_var": torch.exp(self.log_time_var),
            "time_period": torch.exp(self.log_time_period),
            "noise_var": torch.exp(self.log_noise),
        }

    def get_S_from_L(self):
        L = torch.tril(self.L_unconstrained)
        S = L @ L.t()
        S = S + 1e-6 * torch.eye(self.M, device=self.device)
        return S

    def elbo_mc(self, num_mc=4, minibatch_size=None, jitter=1e-6):
        """
        Monte Carlo ELBO estimate:
          E_q(u)[ log p(y | f(u)) ] - KL(q(u) || p(u))
        minibatch_size: if provided, sample random minibatch of data for likelihood estimate.
        Returns: elbo (scalar), info dict
        """
        kern = self.kernel_params()
        Z = self.Z   # [M,3]

        # Build Kuu and its cholesky (M x M) - small
        Kuu = build_Kuu(Z, kern, jitter)
        cholKuu = torch.linalg.cholesky(Kuu + 1e-8 * torch.eye(self.M, device=self.device))
        Kuu_inv = lambda B: torch.cholesky_solve(B, cholKuu)

        # KL
        S = self.get_S_from_L()   # [M,M]
        kl = kl_qp(self.m, S, Kuu, jitter=jitter)

        # minibatch selection
        if minibatch_size is None or minibatch_size >= self.N:
            idx = torch.arange(self.N, device=self.device)
            n_batch = self.N
        else:
            idx = torch.randperm(self.N, device=self.device)[:minibatch_size]
            n_batch = idx.shape[0]

        X_batch = self.X[idx, :]          # [n_batch, 3]
        cov_batch = self.covs[idx, :]     # [n_batch, 2]
        y_batch = self.y[idx]             # [n_batch]

        # Build Kfu for batch -> [n_batch, M]
        Kfu = build_Kfu(X_batch, Z, kern)  # [n_batch, M]

        # Precompute A = Kfu @ Kuu^{-1}  (n_batch x M)
        #Kuu_eye = torch.eye(self.M, device=self.device)
        #Kuu_inv_eye = Kuu_inv(Kuu_eye)
        #A = Kfu @ Kuu_inv_eye   # [n_batch, M]
        A = torch.cholesky_solve(Kfu.T, cholKuu).T   # (Kuu^{-1} Kfu^T)^T = Kfu Kuu^{-1}


        # Sample u ~ q(u) and compute f_mean = A @ u for each MC
        Lq = torch.linalg.cholesky(S + 1e-8 * torch.eye(self.M, device=self.device))
        elbo_ll = 0.0
        eps = 1e-9
        for _ in range(num_mc):
            z = torch.randn(self.M, device=self.device)
            u = self.m + Lq @ z             # [M]
            f_mean = A @ u                  # [n_batch]
            # linear predictor
            # cov_batch columns correspond to [chl, thetao] (assumed standardized)
            lin = self.alpha + (cov_batch @ self.beta) + f_mean
            rate = torch.exp(lin)
            # Poisson log-likelihood (omit factorial)
            logp = (y_batch * torch.log(rate + eps) - rate).sum()
            elbo_ll += logp / num_mc

        # scale minibatch to full dataset
        scale = float(self.N) / float(n_batch)
        elbo = scale * elbo_ll - kl
        return elbo, {"elbo_ll": elbo_ll.item(), "kl": kl.item(), "scale": scale}

    def predict_rate(self, coords_new_np, covs_new_np, num_samples=200, jitter=1e-6):
        """
        Monte Carlo prediction of expected rate at new inputs
        coords_new_np: [Nnew,3], covs_new_np: [Nnew,2]
        """
        Xnew = torch.tensor(coords_new_np, dtype=torch.get_default_dtype(), device=self.device)
        covs = torch.tensor(covs_new_np, dtype=torch.get_default_dtype(), device=self.device)
        kern = self.kernel_params()
        Z = self.Z
        Kuu = build_Kuu(Z, kern, jitter)
        cholKuu = torch.linalg.cholesky(Kuu + 1e-8 * torch.eye(self.M, device=self.device))
        Kuu_inv = lambda B: torch.cholesky_solve(B, cholKuu)
        Kuu_inv_eye = Kuu_inv(torch.eye(self.M, device=self.device))
        Kfu_new = build_Kfu(Xnew, Z, kern)  # [Nnew, M]
        A_new = Kfu_new @ Kuu_inv_eye        # [Nnew, M]
        S = self.get_S_from_L()
        Lq = torch.linalg.cholesky(S + 1e-8 * torch.eye(self.M, device=self.device))

        rates = []
        for _ in range(num_samples):
            z = torch.randn(self.M, device=self.device)
            u = self.m + Lq @ z
            f_mean = A_new @ u   # [Nnew]
            lin = self.alpha + (covs @ self.beta) + f_mean
            rates.append(torch.exp(lin))
        rates = torch.stack(rates, dim=0)  # [num_samples, Nnew]
        rate_mean = rates.mean(dim=0).cpu().detach().numpy()
        rate_p05 = torch.quantile(rates, 0.05, dim=0).cpu().detach().numpy()
        rate_p95 = torch.quantile(rates, 0.95, dim=0).cpu().detach().numpy()
        return rate_mean, rate_p05, rate_p95
    


  


