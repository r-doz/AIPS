# November 20, 2025: spatial example selection

Source: `reports/seeded_monthly_windows_salinity_oldclassifier/11_train_lgcp/seed_1/runs/`.
The comparison covers November 1–28, 2025 (28 days, four weekly models), not the untested November 29–30. November 20 uses the November 15–21 model, with a training cutoff of November 14.

Metrics were recomputed from final `observed` and `predicted` values in each `spatial_predictions.csv`, using `src.models.metrics_lgcp.evaluate_metrics`. Every daily metric includes all 49 retained cells, before interpolation, geographic display masks, edge fading, or colour normalization. Pooled metrics for each of the four weeks were checked against the saved experiment metrics and agree to numerical precision. Weekly pooled RMSE, Wasserstein and F1 are not averages of daily scores.

## Findings

| Metric | November 20 | Median across 28 days | Median across November 15–21 |
| --- | ---: | ---: | ---: |
| Cell MAE | 0.371181 | 0.294270 | 0.271263 |
| Cell RMSE | 1.037200 | 0.674951 | 0.953356 |
| Absolute error of daily total | 13.661513 | 6.192076 | 5.075643 |
| Activity F1 (> 0) | 0.700000 | 0.445055 | 0.400000 |

November 20 is not a median-performance day across all November days. Its observed total is 31 and its predicted total is 17.338487 (44.1% below observed). Cell MAE is 19th of 28 in ascending order; cell RMSE is 23rd and total-count absolute error is 25th.

However, among the five days with nonzero observed daily totals in November 15–21, November 20 has **exactly the median daily cell MAE**:

| Date | Cell MAE, ascending |
| --- | ---: |
| November 18 | 0.246739 |
| November 21 | 0.271263 |
| November 20 | 0.371181 |
| November 17 | 0.373531 |
| November 19 | 0.679472 |

November 15 and 16 have zero observed and predicted totals and are excluded only in this explicitly conditioned comparison. The MAE for each included day still uses all 49 cells, including zero-count cells. Among the 22 nonzero-observation days across November, November 20 is 13th by ascending cell MAE, not the median.

Activity detection provides a separate explanation for visual agreement: November 20 has the highest F1 in its week (0.70), and the sixth-highest across the 28 days. Seven of eight observed active cells are correctly identified (recall 0.875); 12 cells are predicted active (precision 7/12 = 0.583333). This supports agreement in activity location, but does not establish accurate count magnitudes or agreement restricted to the visible map mask.

The November 15–21 week is one of the middle two weeks by pooled cell MAE (0.277455; third-smallest of four, whose numerical median is 0.266794). It is also third-smallest by pooled RMSE and Wasserstein distance, but has the highest daily-total MAE and lowest pooled activity F1 of the four weeks. With four weeks, the numerical median is the average of the middle two values: there is no single median week. It should not be presented as representative across all metrics.

## Suggested wording

“20 November 2025 is shown as an illustrative active-day example. Its cell-level MAE equals the median among the five days with nonzero observed activity in the 15–21 November evaluation window (0.371 AIS counts per cell). Activity locations are captured with an F1 score of 0.70, although the daily total is underestimated (17.34 predicted versus 31 observed).”

This characterization was identified after the day was preferred visually; it is a retrospective description, not a prespecified representative-day selection rule. Do not describe it as the median day of November. For an independently defined month-wide illustration, choose the selection metric and day eligibility rule first and apply them consistently to both months.

## Reproduction

From the repository root:

```bash
.venv/bin/python reports/paper/map_design_61/november_20_representativeness/analyze.py
```

The script regenerates `daily_metrics.csv` and `weekly_metrics.csv`, prints rank/median comparisons, and checks pooled weekly results against the experiment reports.


## Selecting the November 15–21 window across both months

Using the same seed 1 runs plotted in the notebook, November 15–21 is the **only window that belongs to the central pair simultaneously for cell MAE, cell RMSE, and Wasserstein distance**, both among the four November weeks and among all eight May/November weeks. For eight windows it is the fifth-smallest in all three metrics: the upper middle order statistic, sometimes called the upper median. The ordinary numerical median averages the fourth and fifth values; this week is not exactly equal to that numerical median.

| Metric | November 15–21 | Median of eight windows | Position, smallest first | Difference from median |
| --- | ---: | ---: | ---: | ---: |
| Cell MAE | 0.277455 | 0.266794 | 5 / 8 | +4.0% |
| Cell RMSE | 0.941943 | 0.880826 | 5 / 8 | +6.9% |
| Wasserstein distance | 0.144986 | 0.141556 | 5 / 8 | +2.4% |

Within November, the window is third-smallest of four in each metric. Cell MAE and Wasserstein have the same numerical medians as in the eight-window comparison; November's median cell RMSE is 0.823577.

This is a defensible characterization of a window with central **count-error performance**, relevant to observed/predicted count maps. MAE and RMSE evaluate paired cell-day counts; Wasserstein compares the distributions of counts and does not measure spatial alignment. These are related descriptive metrics, not three independent statistical confirmations. Each window is weighted equally and uses its pooled seven-day evaluation values; the comparison is not across seeds or interpolated map pixels.

The qualification matters: this window has the lowest activity F1 (0.482759) of all eight, and the highest daily-total MAE (9.012056) and RMSE (13.529162) among November windows. It is not representative on every metric and is not the best-performing week. November 20's strong daily F1 does not imply strong week-wide F1.

Suggested combined wording:

“November 15–21 was used as an illustrative window with near-median cell-count errors: across the eight May and November seed-1 evaluation windows, it ranks fifth in ascending order for cell-level MAE, cell-level RMSE and Wasserstein distance. Within this window, November 20 has the median cell-level MAE among the five days with nonzero observed activity.”

This is a retrospective justification following visual inspection, not a previously specified selection protocol. It supports the stated, limited characterization; it does not establish that the example represents every aspect of model performance.

`window_centrality.csv` contains the numerical comparisons. To reproduce the central-pair check, load the source `per_window_metrics.csv`, rank `mae_obs`, `rmse_obs`, and `wasserstein` in ascending order, and retain windows ranked 4 or 5 in all three columns (or 2 or 3 for November alone). The unique match in both comparisons is `2025_november_15_21`.
