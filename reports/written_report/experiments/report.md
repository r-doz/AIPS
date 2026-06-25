# Comparison of LGCP, Baseline Models and Window Poisson GLM

## Experimental setting

The target variable is:

```text
ais_vessels_count
```

The main goal is to predict the number of fishing vessels observed in each spatial cell and day.

Two train/test strategies were considered:

### Random-day split

The model is trained on randomly selected days from the year and tested on the remaining days. The split is performed at day level: if a day is assigned to the test set, all spatial cells from that day are assigned to the test set.

This setup evaluates the ability of the model to predict hidden days within a temporal distribution that is partially seen during training.

### Chronological split

The model is trained on the first part of the year and tested on the final part of the year.

This setup is more realistic from a forecasting perspective, because the model has to generalize to future dates that were not observed during training.



# LGCP model

## Basic LGCP

The first model tested was a Sparse Log-Gaussian Cox Process.

The model uses:

```text
x = longitude, latitude, time, thetao, chl
y = ais_vessels_count
```

The predictive structure is:

```text
y_i ~ Poisson(lambda_i)
log(lambda_i) = linear covariate effect + latent spatio-temporal GP
```

The LGCP is therefore a probabilistic spatio-temporal model. It can learn smooth spatial and temporal patterns, but it does not directly use past observed values of the target variable.

Several configurations were tested:

```text
M = 300, steps = 5000, lr = 0.0001
M = 500, steps = 3000, lr = 0.0001
M = 300, steps = 5000, lr = 0.00005
```

The results were broadly similar. The best configuration was:

```text
M = 300, steps = 5000, lr = 0.0001
```

However, the main limitation remained: the model tended to produce predictions fluctuating around an average trend, without fully capturing sharp daily variations.

## LGCP with calendar and fishing covariates

Additional covariates were then added:

```text
is_holiday
is_weekend
fishing_block
```

The best LGCP configuration used:

```text
chl, thetao, is_holiday, is_weekend, fishing_block
```

This improved almost all metrics compared to the basic LGCP. The daily plots also showed that the model was better able to follow daily fluctuations.

The main finding was:

```text
biogeochemical covariates + weekend/holiday effects improved the model substantially;
fishing_block alone gave only a small improvement;
the full combination gave the best LGCP result.
```

The best random-day LGCP result was:

| Metric                      |   Value |
| --------------------------- | ------: |
| Mean log-likelihood per obs | -0.5100 |
| MAE per obs                 |  0.3852 |
| RMSE per obs                |  0.8554 |
| Mean log-likelihood daily   | -5.2450 |
| MAE daily                   |  6.6958 |
| RMSE daily                  |  7.9279 |


# Baseline models

Two simple Poisson baselines were considered.

## Global Mean Poisson

This baseline predicts the same Poisson rate for all test observations:

```text
lambda_hat = mean(y_train)
```

It is a very simple reference model. Its purpose is to check whether more complex models actually improve over a constant average prediction.

## Last Available Poisson

This baseline predicts, for each spatial cell, the last observed value from the previous available day:

```text
lambda_hat(cell, t) = y(cell, previous available day)
```

If no previous value is available, it falls back to the global training mean.

This model is strong because it directly uses the recent observed target value. However, it is not well calibrated probabilistically. In particular, it can obtain good pointwise MAE while performing poorly in log-likelihood, especially when it predicts very low or zero rates before positive observations.

---

# Window Poisson GLM

The Window Poisson GLM models the target as:

```text
y_t ~ Poisson(lambda_t)
```

with input features:

```text
x = covariates at time t + past y values from the same spatial cell
```

The model therefore explicitly uses local temporal persistence.

## W7 model

The first version used a 7-day lag window:

```text
y(t-1), y(t-2), ..., y(t-7)
```

This model was unstable. With low regularization, it obtained good log-likelihood but poor RMSE, especially at daily level.

The best W7 configuration used:

```text
alpha = 1.0
```

However, it was still worse than the best LGCP and clearly worse than the later W1 models.

## W1 model

The best-performing version used only a 1-day lag:

```text
y(t-1)
```

This means that, for each spatial cell and day, the model predicts the current count using the previous observed count in the same cell, together with the covariates of the current day.

The best regularization parameter was:

```text
alpha = 0.05
```

This was the best model overall among the tested approaches.

Best random-day result:

| Metric                      |   Value |
| --------------------------- | ------: |
| Mean log-likelihood per obs | -0.4594 |
| MAE per obs                 |  0.3046 |
| RMSE per obs                |  0.8069 |
| Mean log-likelihood daily   | -3.9914 |
| MAE daily                   |  4.8930 |
| RMSE daily                  |  6.6424 |

---

# Chronological comparison

The chronological split was used to test a more realistic forecasting setting.

The results were:

| Model                            | Mean LL obs |    MAE obs |   RMSE obs | Mean LL daily |  MAE daily |  RMSE daily |
| -------------------------------- | ----------: | ---------: | ---------: | ------------: | ---------: | ----------: |
| Window Poisson GLM W1 alpha=0.05 | **-0.6369** |     0.3669 | **0.8932** |   **-6.9479** | **7.9335** | **10.2569** |
| Global Mean Poisson              |     -0.7822 |     0.4317 |     0.9811 |       -7.2819 |     8.8687 |     10.6921 |
| LGCP                             |     -0.7904 |     0.4301 |     0.9873 |       -7.5264 |     8.8467 |     10.9750 |
| Last Available Poisson           |     -2.3214 | **0.2774** |     0.8952 |      -41.2382 |     8.5135 |     11.1027 |

The Window Poisson GLM W1 alpha=0.05 was the best model on five out of six metrics.

The only exception was pointwise MAE, where the Last Available Poisson baseline performed best. However, this baseline had extremely poor log-likelihood, indicating poor probabilistic calibration.


# Main conclusions

## Random-day split

The best model was:

```text
Window Poisson GLM W1 alpha=0.05
```

This model clearly outperformed LGCP and the baseline models on most metrics, especially at daily level.

## Chronological split

The best model was again:

```text
Window Poisson GLM W1 alpha=0.05
```

This is an important result because the chronological split is more realistic and more difficult than the random-day split.

## LGCP

The LGCP is a meaningful probabilistic spatio-temporal model, and the addition of calendar-related covariates improved its performance.

However, in the current version, LGCP does not directly use past observed target values. This is likely one of the main reasons why it underperforms compared to the Window Poisson GLM.

In the chronological split, LGCP performed similarly to the Global Mean Poisson baseline and did not clearly improve over it.

## Last Available Poisson

The Last Available Poisson baseline is strong in pointwise MAE because it directly uses the previous observed value.

However, it performs poorly as a probabilistic model, especially in terms of log-likelihood. This suggests that it is not well calibrated and can be heavily penalized when it assigns too low a rate to observations with positive counts.

## Overall conclusion

The results suggest that local temporal persistence is a key source of predictive information.

At the current stage, the best-performing approach is:

```text
Window Poisson GLM with W1 and alpha = 0.05
```

The next natural step is to test whether LGCP can be improved by adding an autoregressive covariate:

```text
lag_1 = ais_vessels_count of the previous day in the same spatial cell
```

This would allow a direct comparison between:

```text
LGCP without lag_1
LGCP with lag_1
Window Poisson GLM W1
```
