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

# LGCP multi kernel 

After introducing the multi-kernel LGCP formulation, we tested several temporal kernel configurations to improve chronological generalization on the test set.

The original multi-kernel model fitted the training period well, but during the chronological test period it tended to produce conservative daily predictions close to an average level. This suggests that, when extrapolating in time, the GP component tends to revert toward its prior mean.

The periodic temporal kernel appeared problematic. Since the model uses standardized time, a period value of `1.0` does not necessarily correspond to one year in the model input space. A corrected annual period improved some observation-level metrics, but worsened the daily metrics and produced overly low predictions in the test period. For this reason, we removed the periodic component and tested temporal RBF kernels only.

The temporal RBF experiments were run with fixed lengthscale, initial variance `0.20`, and trainable variance:

| Temporal lengthscale | MAE daily | RMSE daily | Mean LL daily | MAE obs | RMSE obs | Mean LL obs |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 8.4181 | 11.0029 | -8.2891 | 0.3839 | 0.9837 | -0.8075 |
| 0.03 | 8.4805 | 11.0163 | -8.2091 | 0.3840 | 0.9830 | -0.8030 |
| 0.05 | 9.3466 | 11.8589 | -9.5782 | **0.3732** | 0.9846 | -0.8223 |
| 0.10 | **7.2456** | **9.9016** | **-8.0434** | 0.4012 | **0.9615** | **-0.7657** |
| 0.20 | 14.6856 | 19.4433 | -12.2703 | 0.5742 | 1.7102 | -0.8580 |

The best configuration found so far is:

```yaml
temporal_kernels:
  - type: RBF
    hyperparameters:
      lengthscale: 0.10
      variance: 0.20
    trainable:
      lengthscale: false
      variance: true

## Period
Since time is standardized, a periodic kernel with period = 1.0 corresponds to approximately 95 days, not one year. The correct annual period in standardized time is approximately 3.85.

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

## Fixed Test Window Evaluation: December 1–10, 2024

We evaluated the models on a fixed contiguous test window corresponding to the first ten days of December 2024. The training set includes all observations from January 1 to November 30, 2024, while the test set includes only December 1–10, 2024. The remaining dates from December 11 to December 31 were excluded from both training and testing. This setup was designed to evaluate short-term predictive performance on a controlled future window, avoiding the use of the final part of December.

Four models were compared: a Global Mean Poisson baseline, a Last Available Poisson baseline in frozen mode, a Last Available Poisson baseline in one-step-ahead mode, and the multi-kernel LGCP model.

The Global Mean Poisson baseline predicts a constant daily level based on the average training intensity. As expected, it is too rigid: it cannot follow the temporal variation inside the test window. In particular, it underestimates the high-activity days between December 2 and December 6 and overestimates the zero-activity days from December 7 onward.

The Last Available Poisson baseline was evaluated in two modes. In `frozen_train` mode, each cell uses the last observed value before the test window and keeps it fixed throughout December 1–10. This is directly comparable to the LGCP because it does not use any true observations inside the test window. In `one_step_ahead` mode, the model uses the previous observed value even when it belongs to the test set. This represents an online operational baseline, but it is less directly comparable to the LGCP because it uses fresh information from within the test window.

The multi-kernel LGCP achieved the best overall performance. It obtained the best observation-level log-likelihood, observation-level RMSE, daily log-likelihood, daily MAE and daily RMSE. The only metric where it was not the best was observation-level MAE, where the one-step-ahead Last Available baseline performed better. However, this baseline uses previous true test observations and is therefore not a fully frozen multi-day prediction method.

| Model | Mean LL obs | MAE obs | RMSE obs | Mean LL daily | MAE daily | RMSE daily |
|---|---:|---:|---:|---:|---:|---:|
| Global Mean Poisson | -0.7501 | 0.4245 | 0.9057 | -10.1506 | 12.2000 | 13.0099 |
| Last Available Poisson, frozen train | -5.0765 | 0.3837 | 1.1066 | -11.0331 | 12.2000 | 13.6162 |
| Last Available Poisson, one-step-ahead | -2.3591 | **0.3020** | 0.9773 | -50.7814 | 9.2000 | 12.1491 |
| Multi-kernel LGCP | **-0.5420** | 0.3320 | **0.8390** | **-5.6571** | **7.3979** | **9.1199** |

The test-window daily plot confirms the numerical results. During December 1–6, the LGCP correctly predicts a period of positive activity, although it smooths the strongest observed peaks, especially around December 3. On December 7–8, where the observed daily total drops to zero, the LGCP prediction also becomes very low. The main visual error occurs on December 9–10, where the observations remain zero but the model predicts a renewed increase in activity.

Overall, the fixed-window experiment shows that the multi-kernel LGCP adds clear predictive value compared with the baseline models. It is more flexible than the Global Mean baseline, more stable than the Last Available baselines in probabilistic terms, and provides the best daily-level accuracy among the tested models. At the same time, the plot highlights an important limitation: the model tends to smooth sharp peaks and does not always capture sudden drops to zero. This suggests that the LGCP captures the general temporal and spatial structure of the process, but additional information or model refinements may be needed to better predict abrupt changes in fishing activity.


## Fixed Test Window Evaluation: December 1–15, 2024

Model                         LL obs     MAE obs   RMSE obs   LL daily    MAE daily   RMSE daily
Global Mean                  -0.7896     0.4389    0.9455     -9.4323    11.2683     12.8802
Last Available frozen        -5.4411     0.3973    1.1405    -10.5890    11.4667     13.6821
Last Available one-step      -2.9455     0.3347    1.0169    -73.7131    11.4667     13.8756
LGCP multi-kernel            -0.6305     0.3496    0.8967     -6.1605     7.6881     10.1395


## Spatial Kernel Tuning on the Fixed Test Window: December 1–10, 2024

We performed a spatial-kernel tuning analysis on the fixed test window from December 1 to December 10, 2024. The goal was to understand how the spatial component of the LGCP affects both daily-level prediction and observation-level spatial accuracy. In this setup, the model was trained on all observations before the test window, while the test set consisted of 49 spatial cells over 10 days, for a total of 490 observations.

Throughout this tuning phase, the temporal kernel was kept fixed in structure. In particular, we used a temporal RBF kernel with lengthscale equal to `0.10`, corresponding to approximately 9–10 days in the original time scale. The kernel noise was fixed at `1e-3`. Therefore, the experiments focused only on the spatial kernel.

The main question was whether the spatial structure should be represented by a single RBF kernel, a single Rational Quadratic kernel, or a combination of both. Since the objective of the project is not only to predict the total number of vessels per day, but also to predict their spatial distribution, both daily-level and observation-level metrics were considered.

The tested spatial-kernel configurations are summarized below.

| Spatial kernel | Mean LL obs | MAE obs | RMSE obs | Mean LL daily | MAE daily | RMSE daily |
|---|---:|---:|---:|---:|---:|---:|
| RBF + RQ, original setting | -0.5420 | 0.3320 | 0.8390 | -5.6571 | 7.3979 | 9.1199 |
| RQ only | -0.5327 | 0.3458 | 0.8288 | -5.6979 | 7.3698 | 9.1505 |
| RBF only, lengthscale 0.25 | -0.5343 | 0.3315 | 0.8391 | -5.6424 | 7.4822 | 9.0701 |
| RBF only, lengthscale 0.35 | **-0.5121** | **0.3180** | **0.8159** | -5.8175 | 7.7473 | 9.4042 |
| RBF only, lengthscale 0.50 | -0.5267 | 0.3445 | 0.8596 | **-5.1884** | 6.4633 | 7.9756 |
| RBF only, lengthscale 0.65 | -0.5604 | 0.3463 | 0.8516 | -6.0588 | 7.8488 | 9.6871 |
| RBF only, lengthscale 0.75 | -0.5670 | 0.3626 | 0.8853 | -5.7166 | 7.2114 | 9.0522 |
| RBF 0.35 + RQ, RBF lengthscale trainable | -0.5388 | 0.3501 | 0.8390 | -5.5115 | 6.5732 | 8.5262 |
| RBF 0.35 fixed + RQ | -0.5219 | 0.3534 | 0.8226 | -5.3231 | **6.3511** | **7.9519** |

The first important result is that a single RBF kernel with lengthscale initialized at `0.50` performed very well at the daily level. It achieved the best daily log-likelihood and strong daily error metrics. This suggests that an intermediate spatial lengthscale allows the model to capture the overall spatial distribution of intensity in a way that is useful for predicting daily totals.

However, the RBF-only model with lengthscale `0.35` achieved the best observation-level metrics. It obtained the best observation-level log-likelihood, MAE and RMSE among the tested configurations. This indicates that a more local spatial kernel is beneficial for predicting the position of vessels at the cell level. At the same time, this configuration performed worse at the daily level, suggesting that a purely local spatial structure may fragment the intensity field and reduce the quality of aggregate daily predictions.

This revealed a clear trade-off. Shorter spatial lengthscales improved spatial localization, while intermediate lengthscales improved daily totals. Since the final objective of the model is both to estimate the number of vessels and to locate them spatially, neither criterion should be considered alone.

We then tested whether adding a Rational Quadratic kernel to a local RBF kernel could provide a better compromise. The intuition was that the local RBF component could preserve spatial localization, while the Rational Quadratic component could capture broader spatial variation and help recover daily-level accuracy.

When the RBF lengthscale was initialized at `0.35` but left trainable, the model moved the RBF lengthscale from `0.35` to approximately `0.46`. This showed that the model naturally tends to move the RBF component toward the intermediate range around `0.45–0.50`, which is consistent with the strong daily performance of the RBF-only model with lengthscale `0.50`. In the same run, the Rational Quadratic component became relatively broad, with final lengthscale around `2.26`. This improved the daily metrics compared with the purely local RBF model, but did not provide the best overall compromise.

The most promising result was obtained by fixing the RBF lengthscale at `0.35` and adding a trainable Rational Quadratic component. The final learned kernel parameters were:

```yaml
spatial:
  - type: rbf
    params:
      lengthscale: 0.35
      variance: 3.1523

  - type: rational_quadratic
    params:
      alpha: 0.80
      lengthscale: 1.1105
      variance: 2.5845

temporal:
  - type: rbf
    params:
      lengthscale: 0.10
      variance: 1.0699

noise_var: 0.001

## November 21-30 and robustness

To assess whether the spatial-kernel tuning performed on December 1–10 was robust, we evaluated the two main candidate kernels on an additional fixed test window, November 21–30, 2024. This second window revealed an important difference between the two configurations. The combined kernel with fixed local RBF and Rational Quadratic components, which performed very well on December 1–10, degraded substantially on the November window. In particular, it strongly overestimated the daily totals between November 25 and November 29, leading to a daily RMSE of 19.72.

By contrast, the simpler RBF-only kernel with initial lengthscale 0.50 remained much more stable. On the November 21–30 window, it achieved a daily MAE of 7.36 and a daily RMSE of 9.55, substantially improving over the combined kernel. It also outperformed the frozen Last Available baseline in observation-level log-likelihood, observation-level RMSE, daily log-likelihood and daily MAE, while remaining close in daily RMSE.

This suggests that the combined RBF + Rational Quadratic kernel may have been partially tuned to the December 1–10 window and is not yet robust across different future periods. The RBF-only kernel with lengthscale 0.50 provides a better compromise between accuracy, robustness and model simplicity. Therefore, despite the strong December performance of the combined kernel, the current preferred spatial kernel is the simpler RBF-only configuration.

Model                         LL obs     MAE obs   RMSE obs   LL daily    MAE daily   RMSE daily
LGCP RBF0.35 fixed + RQ       -0.7250    0.5974    1.4888     -9.4859     16.0702     19.7201
LGCP nuova run                -0.5729    0.4187    0.7874     -5.3659      7.3569      9.5475
Last Poisson one-step         -2.6723    0.2857    0.8157    -16.3761      6.8000      9.1104
Last Poisson frozen           -3.5322    0.3796    1.0361     -5.9212      7.8000      8.7636