# Activity classification metrics

GNN, global_cell_mean, last_available_poisson, window-GLM, LGCP, and
ConvLSTM report `accuracy`, `precision`, and `recall` alongside the existing
count metrics in `metrics.yaml` and console output. Baseline comparison CSVs
also include them; the eight-window runner aggregates them automatically.

Following Fishes 2025, 10, 479, pp. 9–10, activity means a value strictly
greater than zero. Each evaluated cell-day is one observation: the observed
count and final predicted rate are independently converted to active/inactive
labels. Scores are pooled over observations in the evaluated horizon, rather
than averaged over days. ConvLSTM uses its existing valid-cell mask.

- Accuracy = (TP + TN) / (TP + TN + FP + FN).
- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN).

Scores are fractions from 0 to 1. Undefined precision or recall is reported as
0; empty inputs yield NaN. Exact zero predictions remain inactive even when
the likelihood calculation uses a positive numerical floor.

This rule uses predicted counts/rates, not classifier probabilities or rounded
counts. A model predicting positive rates everywhere will have recall 1 when
observed activity exists, and accuracy and precision equal the observed active
fraction. These scores measure activity detection, not exact count agreement.

New evaluations write the additional metrics. Existing reports are not
automatically recalculated.

## Wasserstein distance

The same models also report `wasserstein` in console summaries, `metrics.yaml`,
and baseline comparison CSVs. Eight-window summaries aggregate it automatically;
LGCP hyperparameter sweeps treat it as a metric to minimize.

Following Table 2, Equation 9 of [Agmata and Guðmundsson (2025), CATCH](https://doi.org/10.1093/biomethods/bpaf045),
the one-dimensional distance is `W1 = integral |CDF_obs(x) - CDF_pred(x)| dx`.
We use equally weighted empirical distributions of observed counts and final
predicted rates, pooling valid cell-day observations over the evaluated horizon.
SciPy's `wasserstein_distance` computes the integral directly from these samples.
Likelihood-specific epsilon floors are excluded.

Lower is better; zero means identical value distributions. Results are in count
units, adapting the paper's probability-density predictions to this project's
count predictions without normalizing daily totals. The metric ignores cell/day
ordering: permuting predictions leaves it unchanged. It therefore measures
distributional agreement, not geographical transport distance. Empty or
non-finite inputs yield NaN. Existing reports require reevaluation.

## Trend metrics and first-three-day reports

LGCP, GNN, Window-GLM, ConvLSTM, global cell mean (including expanding mode),
and last available (including one-step-ahead mode) save all 12 metrics:
`mean_ll_obs`, `mae_obs`, `rmse_obs`, `wasserstein`, `mean_ll_daily`,
`mae_daily`, `rmse_daily`, `daily_delta_corr`,
`daily_direction_accuracy_moving`, `accuracy`, `precision`, and `recall`.
The two trend metrics also appear in console summaries. Correlation compares
changes in daily totals; direction accuracy excludes flat observed transitions.

When the test window contains more than three distinct calendar dates, these
scripts also save `metrics_first_3_days.yaml`. It contains the first three dates,
the observation count, `aggregation: pooled_first_three_test_days`, and all
metrics evaluated on the corresponding final predictions. This does not retrain
models or change their forecast mode. Undefined values are stored as YAML null;
in particular, three days provide too few transitions for the correlation
implementation. Windows of three or fewer dates use only `metrics.yaml`.
Global cell mean retains its existing epsilon convention for likelihoods in
both reports; trend, activity and Wasserstein use predictions before that floor.
Existing saved experiments are not automatically updated.

## Local classifier redistribution (LGCP)

To select local redistribution in an LGCP config, use:

```yaml
zero_gate:
  enabled: true
  mode: local_redistribuite # local_redistribute is also accepted
  threshold: 0.3
  local_radius: 1 # Chebyshev distance in spatial grid steps
```

Keep the other classifier settings as desired. This mode uses longitude/latitude
cell centres converted to regular grid indices, not standardized model features.
For each rejected cell, its predicted rate is transferred to classifier-accepted
cells on the same date within the specified radius. Radius 1 includes horizontal,
vertical and diagonal neighbours. If none exist within the radius, all accepted
cells tied at the nearest Chebyshev distance receive the rate instead.
Shares are proportional to recipients' original LGCP rates; if all recipient
rates are zero, shares are equal. Weights never use already redistributed rates,
so rejected-cell processing order does not change the result.

Daily predicted totals are preserved. If the classifier accepts no cells on a
date, original predictions for that date are retained with a warning, matching
the existing global redistribution fallback. Existing configs keep their chosen
mode; enabling this option does not require changing the kernel configuration.

## Confidence-weighted classifier redistribution (LGCP)

```yaml
zero_gate:
  enabled: true
  mode: confidence_redistribute
  threshold: 0.2
  redistribution_gamma: 1.0
  redistribution_scale: 1.0
```

For a rejected cell with LGCP rate `r`, classifier probability `p`, and
threshold `t`, this mode recovers `r * redistribution_scale * p / t`. The
unrecovered rate is discarded. Within each date, recovered mass is assigned to
accepted cells using weights `r * p ** redistribution_gamma`. It therefore
does not force preservation of an unreliable LGCP daily total. If a date has
no accepted cells, hard-gate behavior is retained and its predictions are zero.
