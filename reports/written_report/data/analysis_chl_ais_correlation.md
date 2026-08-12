# Chl / ais correlation analysis 

## Chlorophyll–Vessel Correlation Analysis

After the experiments with vessel-count lags, we investigated the relationship between chlorophyll concentration and AIS vessel activity. This analysis was motivated by visual comparisons between chlorophyll and vessel activity, which suggested that periods of higher chlorophyll often correspond to periods of higher fishing activity.

We computed lagged correlations between the daily total number of AIS vessels and the daily mean chlorophyll concentration in the Gulf of Trieste. The convention used was:

```text
corr(vessels_t, chl_{t-lag})