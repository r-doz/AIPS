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
