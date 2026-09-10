"""Log-linear NB-INGARCH-X for one-step-ahead daily count forecasts.

eta[t] = intercept + X[t] @ beta + sum(a[j] * log1p(y[t-lag[j]]))
         + b * eta[t-1]. Positive dynamic coefficients sum to less than 0.98.
The first max(lags) observations initialize the recursion and are not scored.
"""

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import digamma, expit, gammaln, softmax


class NBINGARCH:
    def __init__(self, observation_lags=(1, 7), ridge=0.1, max_iter=1000):
        self.lags = tuple(observation_lags)
        if (not self.lags or any(not isinstance(v, int) or isinstance(v, bool) or v < 1
                                 for v in self.lags) or len(set(self.lags)) != len(self.lags)):
            raise ValueError("observation_lags must be distinct positive integers")
        if not np.isfinite(ridge) or ridge < 0 or max_iter < 1:
            raise ValueError("ridge must be nonnegative and max_iter positive")
        self.ridge, self.max_iter = float(ridge), int(max_iter)

    def _validate(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X and y must be aligned daily arrays")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Inputs must be finite")
        if (y < 0).any() or not np.allclose(y, np.round(y), rtol=0, atol=1e-8):
            raise ValueError("Daily targets must be nonnegative integer counts")
        if len(y) <= max(self.lags):
            raise ValueError("Not enough days for the configured observation lags")
        return X, y

    def _design(self, y):
        start = max(self.lags)
        return np.column_stack([np.log1p(y[start-lag:len(y)-lag]) for lag in self.lags])

    def _filter(self, params, X, lagged, gradient=False):
        p = X.shape[1]
        probabilities = softmax(np.r_[params[1+p:-1], 0.0])
        weights = 0.98 * probabilities[:-1]
        a, b = weights[:-1], weights[-1]
        base = params[0] + X @ params[1:1+p] + lagged @ a
        eta = lfilter([1.0], [1.0, -b], base, zi=[b*self.initial_eta_])[0]
        if not gradient:
            return eta
        previous = np.r_[self.initial_eta_, eta[:-1]]
        weight_jac = 0.98 * (np.diag(probabilities[:-1])
                            - np.outer(probabilities[:-1], probabilities[:-1]))
        direct = np.column_stack([np.ones(len(X)), X,
                                  np.column_stack([lagged, previous]) @ weight_jac])
        derivative = lfilter([1.0], [1.0, -b], direct, axis=0)
        return eta, derivative

    def _objective(self, params, X, y, lagged):
        eta, derivative = self._filter(params, X, lagged, gradient=True)
        log_k = params[-1]
        k = np.exp(log_k)
        log_sum = np.logaddexp(log_k, eta)
        nll = (gammaln(k) + gammaln(y+1) - gammaln(y+k)
               + k*(log_sum-log_k) + y*(log_sum-eta))
        d_eta = (k+y)*expit(eta-log_k)-y
        d_k = (digamma(k)-digamma(y+k) + log_sum-log_k-1
               + (1+y/k)*expit(log_k-eta))
        beta = params[1:1+X.shape[1]]
        grad = np.r_[derivative.T @ d_eta / len(y), np.mean(d_k)*k]
        grad[1:1+X.shape[1]] += self.ridge*beta
        return float(nll.mean() + 0.5*self.ridge*(beta @ beta)), grad

    def fit(self, X, y):
        X, y = self._validate(X, y)
        self.n_features_ = X.shape[1]
        self.initial_eta_ = float(np.log(max(y.mean(), 1e-3)))
        params = np.zeros(1+X.shape[1]+len(self.lags)+1+1)
        params[0] = self.initial_eta_ * (1 - 0.98*(len(self.lags)+1)/(len(self.lags)+2))
        params[-1] = np.log(10.0)
        start = max(self.lags)
        result = minimize(self._objective, params,
                          args=(X[start:], y[start:], self._design(y)), jac=True,
                          method="L-BFGS-B",
                          bounds=[(None, None)]*(len(params)-1)+[(-8, 16)],
                          options={"maxiter": self.max_iter, "ftol": 1e-10})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("NB-INGARCH optimization produced nonfinite values")
        if not result.success:
            warnings.warn(f"NB-INGARCH optimizer did not converge: {result.message}", RuntimeWarning)
        self.params_ = result.x
        self.dispersion_ = float(np.exp(result.x[-1]))
        self.fit_info_ = {"converged": bool(result.success), "message": str(result.message),
                          "iterations": int(result.nit), "penalized_nll": float(result.fun)}
        return self

    def predict_one_step(self, X, observed_y):
        """Filter a full history from its training start with frozen parameters.

        Each prediction uses only counts strictly before its date. Call with the
        training prefix followed by test observations; this is not a forecast
        of an entire test window made at a single origin.
        """
        if not hasattr(self, "params_"):
            raise ValueError("Fit the model before prediction")
        X, y = self._validate(X, observed_y)
        if X.shape[1] != self.n_features_:
            raise ValueError("Prediction feature count differs from training")
        eta = self._filter(self.params_, X[max(self.lags):], self._design(y))
        means = np.exp(eta)
        if not np.isfinite(means).all() or (means <= 0).any():
            raise RuntimeError("Forecast means must be finite and positive")
        return np.r_[np.full(max(self.lags), np.nan), means]
