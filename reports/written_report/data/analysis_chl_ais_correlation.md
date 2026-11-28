# Chl / ais correlation analysis 

## Chlorophyll–Vessel Correlation Analysis

After the experiments with vessel-count lags, we investigated the relationship between chlorophyll concentration and AIS vessel activity. This analysis was motivated by visual comparisons between chlorophyll and vessel activity, which suggested that periods of higher chlorophyll often correspond to periods of higher fishing activity.

We computed lagged correlations between the daily total number of AIS vessels and the daily mean chlorophyll concentration in the Gulf of Trieste. The convention used was:

```text
corr(vessels_t, chl_{t-lag})
```

With this convention, positive lags mean that chlorophyll precedes vessel activity, while negative lags mean that vessel activity precedes chlorophyll.

The strongest positive correlations occurred at weekly multiples:

| Lag | Interpretation             | Pearson correlation | Spearman correlation |
| --: | -------------------------- | ------------------: | -------------------: |
|   0 | same-day chlorophyll       |              0.4147 |               0.4598 |
|   7 | chlorophyll 7 days before  |              0.4477 |               0.5032 |
|  14 | chlorophyll 14 days before |              0.4759 |               0.5246 |
|  21 | chlorophyll 21 days before |              0.4977 |               0.5219 |
|  28 | chlorophyll 28 days before |              0.5130 |               0.5058 |

These results show a clear positive association between chlorophyll concentration and AIS vessel activity. The association is not only contemporaneous: chlorophyll measured in previous weeks is also strongly associated with later vessel activity. This suggests that chlorophyll contains useful predictive information for fishing-vessel presence.

We also analysed the relationship between daily changes in vessel activity and daily changes in chlorophyll. The correlations between changes were also strong:

| Lag | Interpretation                    | Pearson correlation |
| --: | --------------------------------- | ------------------: |
|   0 | same-day changes                  |              0.5170 |
|   7 | chlorophyll change 7 days before  |              0.5168 |
|  14 | chlorophyll change 14 days before |              0.5200 |
|  21 | chlorophyll change 21 days before |              0.5343 |
|  28 | chlorophyll change 28 days before |              0.5213 |

This suggests that not only the absolute level of chlorophyll, but also its temporal variation, may be related to changes in vessel activity.

An additional ablation supported the importance of environmental covariates. When chlorophyll and temperature were removed from the LGCP and only cell_lag_1 was used as additional information, performance degraded substantially on the November 21–30 window.

| Model                                                   | Mean LL obs | MAE obs | RMSE obs | Mean LL daily | MAE daily | RMSE daily |
| ------------------------------------------------------- | ----------: | ------: | -------: | ------------: | --------: | ---------: |
| LGCP with chlorophyll and temperature                   |     -0.5729 |  0.4187 |   0.7874 |       -5.3659 |    7.3569 |     9.5475 |
| LGCP with `cell_lag_1`, without chlorophyll/temperature |     -0.6863 |  0.5503 |   1.1394 |       -9.6284 |   16.0973 |    19.2581 |


This indicates that chlorophyll and temperature are not marginal covariates: they provide substantial information for the LGCP. In contrast, raw vessel-count lag features alone were not able to recover good predictive performance.

The current LGCP already uses cell-level chlorophyll chl(s,t) as a covariate. This provides local spatial information and can help the model decide where to place vessel intensity. However, the correlation analysis suggests that global daily chlorophyll summaries may also be useful, especially for calibrating the total daily level of fishing activity.

Based on these diagnostics, the next modelling step is to enrich the chlorophyll representation by adding daily-scale chlorophyll features. The most natural candidates are:

daily_mean_chl: mean chlorophyll over the whole Gulf at day t;
daily_chl_diff: difference between daily mean chlorophyll at day t and day t-1;
daily_mean_chl_lag_7: daily mean chlorophyll one week before;
daily_mean_chl_lag_14: daily mean chlorophyll two weeks before.

The rationale is that cell-level chlorophyll captures local spatial productivity, while daily mean chlorophyll captures the global environmental regime of the Gulf. The difference feature captures recent changes in chlorophyll, while the lagged features are motivated by the observed weekly-lag correlation structure.

These correlations should not be interpreted as causal evidence, because chlorophyll and vessel activity may both be influenced by seasonal, weekly, meteorological or operational factors. However, they provide strong evidence that chlorophyll contains useful predictive information and is a more promising direction than raw vessel-count lags for improving the LGCP.