# November 20, 2025: justification across all evaluated days

**November 20 is a defensible illustrative day with broadly moderate errors and good activity detection. It is not uniquely the most representative day.** The justification below compares it directly with all 56 evaluated days; no week-level selection argument is needed.

## Scope

The analysis uses notebook 62’s seed-1 hurdle predictions (`pred_lower`), covering May 1–28 and November 1–28, 2025. Each daily metric uses all 49 retained cells, including zero-count cells, before interpolation, geographic display masks or edge fading. These statements concern the underlying evaluation grid, not exclusively the visible portion of the map. The 56 days receive equal weight in daily summaries.

## 1. Consistent central-range evidence across several metrics

A single common rule avoids choosing a different favourable reference for each metric: compare each daily score with the **middle 50% (25th–75th percentiles)** of scores across all 56 days.

| Metric | November 20 | Middle 50% of daily values | Midrank percentile |
| --- | ---: | ---: | ---: |
| Cell MAE | 0.338823 | 0.082355–0.419981 | 52.7 |
| Cell RMSE | 0.953170 | 0.391192–1.060239 | 72.3 |
| Wasserstein distance | 0.270044 | 0.061224–0.274436 | 74.1 |
| Absolute daily-total error | 6.210561 | 1.214352–9.228301 | 58.0 |
| Activity F1 | 0.700000 | 0.000000–0.701471 | 73.2 |

**All five values are inside their respective middle-50% intervals.** Ten of the 56 days satisfy this simultaneous condition, including November 20. The criterion covers paired count errors, count-distribution mismatch, daily-total error and activity detection. It supports a reasonably central example without claiming a unique selection. The metrics are correlated, so this is a descriptive check, not five independent statistical confirmations.

RMSE and Wasserstein are near the upper edge of the central range, whereas MAE and daily-total error are closer to its centre. Higher F1 means better performance; higher error scores mean worse performance. Daily-total MAE and RMSE coincide for a single day, so they are counted only once.

## 2. Other standard aggregators

| Metric | November 20 | Mean of daily scores | Median of daily scores | 10% trimmed mean | Pooled metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cell MAE | 0.338823 | 0.291212 | 0.331830 | 0.286469 | 0.291212 |
| Cell RMSE | 0.953170 | 0.765769 | 0.746648 | 0.726428 | 0.930981 |
| Wasserstein | 0.270044 | 0.190657 | 0.173682 | 0.176554 | 0.089014 |
| Absolute daily-total error / aggregate daily MAE | 6.210561 | 6.737869 | 5.178738 | 5.701297 | 6.737869 |
| Activity accuracy | 0.877551 | 0.891035 | 0.877551 | 0.893966 | 0.891035 |
| Activity precision | 0.583333 | 0.426487 | 0.428571 | 0.410506 | 0.557143 |
| Activity recall | 0.875000 | 0.505049 | 0.563492 | 0.506147 | 0.674352 |
| Activity F1 | 0.700000 | 0.437067 | 0.500000 | 0.435858 | 0.610169 |

The clearest additional facts are:

- **Cell MAE is 2.11% above the daily median** (30th of 56 in ascending order).
- **Activity accuracy exactly equals the daily median: 43/49 = 87.76%.** The observed/predicted activity confusion counts are 7 true positives, 36 true negatives, 5 false positives and 1 false negative. Accuracy is influenced by the many inactive cells; precision and recall provide the necessary context.
- **Cell RMSE is 2.38% above the pooled cell RMSE**, 0.953170 versus 0.930981. This is closeness to overall squared-error performance, not to the median daily RMSE: it is 27.66% above that daily median.
- **Daily-total error is 7.83% below the mean daily-total error**, 6.210561 versus 6.737869. It is 19.92% above the daily median and 8.93% above the trimmed mean.
- **F1 = 0.70 and recall = 0.875 show good activity detection**, rather than median detection performance. Seven of eight observed active cells are detected; twelve cells are predicted active. Recall is around the 81st percentile, outside the middle 50%.

The pooled column evaluates the concatenated cell-day records (or their daily totals). With 49 cells every day, pooled cell RMSE equals `sqrt(mean(daily_RMSE**2))`, not `mean(daily_RMSE)`. Pooled Wasserstein compares the concatenated count distributions; pooled F1 uses combined confusion counts. Neither is a mean of daily scores. The trimmed mean removes the lowest and highest `floor(0.1*n)` daily values separately for each metric (five at each end for n=56).

The full table matters: choosing only the closest reference separately for each metric would overstate representativeness. In particular, Wasserstein is higher than its mean, median, trimmed mean and pooled value. November 20 is not near the centre under every aggregator.

## 3. Combined summaries across metrics

As a sensitivity check, metrics are equally weighted after putting them on comparable scales. Lower scores mean closer to the centre, not better predictive performance. No weights were fitted to favour November 20.

| Metric set | RMS standardized distance from means: rank / 56 | Mean absolute distance from medians, scaled by IQR: rank / 56 | Mean percentile distance from 50th percentile: rank / 56 |
| --- | ---: | ---: | ---: |
| Cell MAE, cell RMSE, daily-total error | 6 | 7 | 9 |
| Above + Wasserstein | 11 | 12 | 13.5 |
| Above + activity F1 | 10 | 14 | 13.5 |

Using the broad five-metric set, November 20 lies approximately within the **most central quarter of days** under all three summaries. This reinforces moderate overall centrality, but does not make it the most central day. These are exploratory summaries: metric inclusion, scaling and correlations affect the ranking. They should not be presented as a prespecified selection algorithm.

For reproducibility, the scores are:

- RMS standardized distance: `sqrt(mean(((score_j - mean_j) / sample_std_j)**2))`.
- Median/IQR distance: `mean(abs(score_j - median_j) / (q75_j - q25_j))`.
- Mean percentile distance: `mean(abs((average_rank_j - 0.5) / n - 0.5))`.

The exports also include maximum percentile distance, which asks whether any included metric is far from the centre. In the broad five-metric set, November 20 ranks ninth by this criterion. The three metric sets and all four aggregators examined are retained in the CSV, rather than reporting only the most favourable combination. Tied scores are rounded to 12 decimal places before average ranks are assigned.

## 4. Active-day sensitivity, without conditioning on a week

Eleven of the 56 days have zero observed activity. As a secondary comparison, restricting to the **45 days with positive observed totals across both months** gives:

- Cell MAE 0.338823 versus mean 0.346369 (−2.18%) and median 0.356430 (−4.94%).
- Daily-total absolute error 6.210561 versus median 6.188321 (+0.36%), ranked 24th of 45.
- Activity precision 0.583333 equals the active-day median.
- Eight observed active cells equals the median number of active cells.
- The same five main metrics remain inside their respective middle-50% intervals.
- Combined five-metric centrality ranks are third by standardized mean distance, seventh by median/IQR distance, and fifth by mean percentile distance, out of 45.

This is useful if the figure is explicitly described as an **active-day illustration**, but the main justification above retains all 56 days. It is a different reference population and must be named if used. Daily MAE still includes all 49 cells.

## Suggested wording, with no week reference

“20 November 2025 is shown as an illustrative active day. Across the 56 evaluated days, its cell-level MAE (0.339 counts per cell) is close to the median (0.332), and its cell MAE, cell RMSE, Wasserstein distance, daily-total absolute error and activity F1 all fall within their respective interquartile ranges.”

Optional additional sentence:

“Its activity-classification accuracy equals the median daily accuracy (87.8%), while its cell RMSE (0.953) is close to the pooled evaluation RMSE (0.931).”

The observed total is 31 and the predicted total is 24.79 (20.0% underestimation). The observed total is above the overall upper quartile of 29, so the day is relatively busy. These findings support an illustrative day with moderate error scores and good activity detection, not typical traffic volume or representativeness on every metric. All selection evidence is retrospective because the date was already chosen.

## Reproduction and validation

From the repository root:

```bash
.venv/bin/python reports/paper/map_design_62/november_20_representativeness/analyze.py
```

The script reloads the exact seed-1 source CSVs, checks their dates, variant, gate settings, unique cell-day records and finite nonnegative values, and reproduces all eight saved pooled metric reports within numerical tolerance. The source paths are in `sources.csv`.

New outputs:

- `across_days_aggregates.csv`: means, medians, trimmed means, quartiles, pooled values and percentiles for all days and observed-active days.
- `multimetric_centrality.csv`: every date’s score and rank for every metric set and aggregator.
- `joint_middle_50pct.csv`: which days satisfy the common interquartile-range criterion.

The earlier `daily_metrics.csv`, `weekly_metrics.csv`, `day_centrality.csv` and `window_centrality.csv` remain reproducible for audit purposes. The justification above does not use weekly ranks. The saved Poisson log-likelihood scores are not used as hurdle likelihoods, and temporal-trend scores are not single-day performance metrics.
