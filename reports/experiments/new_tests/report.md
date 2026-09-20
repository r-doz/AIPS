# Validation of LGCP on 1-7/11/2025

The original model was spatial RBF + temporal RBF + temporal periodic. With temporal RBF with l = 0.08, while spatial RBF with l = 0.2.

## Temporal: periodic vs no periodic 
I tested temporal without periodic. 

It seems that the model with also the temporal periodic component works better on almost every metrics.

## Temporal: lengthscale

### Get a good one 
I tested the original model with temporal RBF l = [0.08, 0.03, 0.02, 0.01, 0.05, 0.12, 0.005]

It seems 0.02 and 0.01 are the strongest componets on the daily metrics and good overall. While, 0.12 + classificator Active is the best on Observation metrics, but worse on the other. 

### Test the extreme 

After we saw 0.01 is good overall, I tested some extremes: really smaller than 0.01 to imrpove the daily metrics, while really higher than 0.01 to improve the OBS level. I tried a temporal RBF lengthscale of [0.01, 0.001, 0.2, 0.003, 0.15, 0.25].

The bests one overall were [0.01, 0.001, 0.003]. Looking in detail, the best one is [0.01].

## Spatial: lengthscale

I tested the original model with spatial RBF l = [0.2, 0.1, 0.5, 0.05, 0.15] (with temporal RBF l = 0.01)

It seems that the results are almost the same, with small differences. 0.1 is almost the best on all metrics but one, in which 0.5 was a bit bit better, but super worse in the others. 

## Classifier 
I tried 3 classifier, the first with the previous setting and threshold = 0.03 and 0.05. And the classifier with current settings with threshold 0.03. 

- The best overall (apart from a little decrease in correlation) is threshold 0.03 and spatial rbf l = 0.1, temporal rbf l = 0.01.
- In general, adding the classifier become weaker overall but super strong on MAE OBS, by far better than even the GNN. So, it interesting to consider this model, but is not the best one.

## Different classifier and spatial kernels
Since decreasing the threshold means improvement for the classifier, another test could be done by decreasing the trheshold from 0.3.  

Another class of test regards the space intensity. One visual problem is that our model is prudent, the forcast tend to be wide but with small values. So, some tests have to be done:
- same setting, without classifier, with space lengscale 0.1, variance = 4, false, false
- same setting, classifier, with space lengscale 0.1, variance = 6, false, false
- add a further spatial RBF, so: RBF(0.2, 2, F, T) + RBF(0.03, 0.5, F, T) + classifier
- add a further spatial RQ: so, RBF(0.1, 2, F, T) + RQ(0.05, 0.5, 1, F, T, T) + classifier
- same setting, NC, with space lengscale 0.1, variance = 6, false, false
- add a further spatial RBF, so: RBF(0.2, 2, F, T) + RBF(0.03, 0.5, F, T), NC 
- add a further spatial RQ: so, RBF(0.1, 2, F, T) + RQ(0.05, 0.5, 1, F, T, T), NC 

Results:

In terms of all metrics but the MAE-OBS the best models are
- Non-classifier 2 spatial RBF (slightly better than the one below on all metrics)
- NC, rbf l = 0.1, temporal rbf l = 0.01 
- NC, 2 spatial kernels: RBF + RQ

In terms of MAE-OBS, the best models are 
- Classifier, rbf l = 0.1, temporal rbf l = 0.01
- Classifier var = 6 and false
- Classifier 2 spatial RBF
- Classifier Spatial RBF + RQ
- (a bit worse than the other above) Classifier with threshold = 0.1

In terms of correlation, the best model are 
- NC, rbf l = 0.1, temporal rbf l = 0.01 
- NC, var = 4 and false
- NC, 2 spatial RBF
- NC var = 6
- NC spatial RBF + RQ
- Classifier with t = 0.1 

Conclusion: 
- The best model overall is NC, 2 spatial RBF, it follows simple NC rbf l = 0.1, temporal rbf l = 0.01 
- Only classifiers are strong on MAE-OBS, with classifier 2 spatial RBF slightly better  
- NC, 2 spatial RBF has a good plot 
- Classifier t = 0.1 is interesting because it is strong both on correlation and MAE-OBS (the plot is good moreover)

## Tune the 2 spatial RBF 
Since the model with 2 spatial RBFs seems to be the stronger on all metrics but MAE-OBS, i tried different configurations:
- lgcp_spat_2rbf_l0.20_l0.03 (original one)
- lgcp_spat_2rbf_l0.10_l0.03
- lgcp_spat_2rbf_l0.30_l0.03
- lgcp_spat_2rbf_l0.20_l0.01
- lgcp_spat_2rbf_l0.20_l0.06
- lgcp_spat_2rbf_trainable_ls 

Results:
There are 3 models which are superior on all the other. These are 
- lgcp_spat_2rbf_l0.20_l0.01 (by a little the best)
- lgcp_spat_2rbf_l0.20_l0.03 (original one)
- lgcp_spat_2rbf_l0.20_l0.06

All the other models are worse, especially the one with lenghscale trianable 

## Tune the single spatial RBF again 
After discovering the best 2 spatial RBF is 
- lgcp_spat_2rbf_l0.20_l0.01
I tried the single spatial RBF with 
- lgcp_spat_2rbf_l0.20_l0.01 (best 2 spatial RBF)
- lgcp_spat_rbf_l0.10 (best 1 spatial RBF)
- lgcp_spat_rbf_l0.20_lr0001 (original lr)
- lgcp_spat_rbf_l0.20_lr001 (new lr)

Results: 
There are two dominant models 
- lgcp_spat_2rbf_l0.20_l0.01 (by a little better)
- lgcp_spat_rbf_l0.10

## Soft classifier 
The studied classifier was of the HARD version. The best found so far is 
- Classifier, rbf l = 0.1, temporal rbf l = 0.01
The best threshold are t = 0.1 and 0.3. With 
- t = 0.3 
Is the best in terms of MAE-OBS
while t = 0.1, is better overall but weaker in MAE compared to t=0.3 (so I chose t = 0.3 since the goal of the classifier is to perform on MAE-OBS).

Tests:
- Classifier, rbf l = 0.1, temporal rbf l = 0.01, t = 0.3
- Classifier, rbf l = 0.1, temporal rbf l = 0.01, t = 0.1
- Classifier, rbf l = 0.1, temporal rbf l = 0.01, SOFT

Results 
The two HARD classifiers 
- Classifier, rbf l = 0.1, temporal rbf l = 0.01, t = 0.3
- Classifier, rbf l = 0.1, temporal rbf l = 0.01, t = 0.1
remains stronger compared to the soft one, so i would discard it.

## Try to make the model "better recovery of observed peaks"
### First trial
The best model so far is 
- 2 RBF spatial 0.2, 0.01 and temporal 0.01
The candidate models are 
- Strengthen the local component: short spatial variance fixed at 0.5
- Increase spatial resolution of the approximation: M = 500
- Allow faster temporal changes: temporal lengthscale fixed at 0.003

Results:
The best model is still 
- 2 RBF spatial 0.2, 0.01 and temporal 0.01
A bit worse but still good is 
- M = 500
The only model that seems to better recover the peaks is 
- M = 500
but nothing special. 

### Second trial 
The best model so far is 
- 2 RBF spatial 0.2, 0.01 and temporal 0.01
I tried 
- Sp 2 RBF l 0.35, l 0.01
- Sp RBF + RQ, RBF l = 0.20, RQ l = 0.30, alpha = 0.8
- Sp RBF + RQ, RBF l = 0.20, RQ l = 1.20, alpha = 0.8  
- Sp RBF + RQ, RBF l = 0.35, RQ l = 1.20, alpha = 0.8  

Results:
There isn't a model with a visible peak-recovery from the plots.
The best model all the metrics is still 
- 2 RBF spatial 0.2, 0.01 and temporal 0.01
Two mention are for 
- Sp 2 RBF l 0.35, l 0.01 
which is very similar but worse, and 
- Sp RBF + RQ, RBF l = 0.20, RQ l = 0.30, alpha = 0.8
whichi is the onlu with the same MAE-OBS equal to 2 RBF spatial 0.2, 0.01 and temporal 0.01

## Improve the spatial 2 RBF 
I tried the best model 
- 2 RBF spatial 0.2, 0.01 and temporal 0.01
against two models with the same setting, but 
- lr 0.005 (instead of 0.001)
- temporal periodic RBF with lengthscale NON-trainbale 

Results:
- The model "temporal periodic RBF with lengthscale NON-trainbale fixed at 0.2" is better or equal to "2 RBF spatial 0.2, 0.01 and temporal 0.01" on all metrics, but MAE-OBS (0.30 vs 0.32). Moreover, the plot shows more peak predictions.
- The lr 0.005 makes a worse results on all the metrics 

Comment:
The temporal periodic lengthscale controls how similar two times within a cycle must be. In other words, if l is high two days within the week (the period) should be similar, while with l low, two days within the week could be really different. 

## Tune the periodic lengthscale 

### First experiment 

We discovered the "spatial 2 RBF, temporal periodic lengthscale fixed at l0.2" outperforms "spatial 2 RBF" across all metrics except for MAE-OBS. 
So, we tried to tune the fixed temporal periodic lengthscale. 
We compared 6 models:
- spatial 2 RBF 0.2, 0.01, periodic l trianable (it reaches 1.6) 
- spatial 2 RBF 0.2, 0.01, periodic l 0.05
- spatial 2 RBF 0.2, 0.01, periodic l 0.1
- spatial 2 RBF 0.2, 0.01, periodic l 0.2
- spatial 2 RBF 0.2, 0.01, periodic l 0.4
- spatial 2 RBF 0.2, 0.01, periodic l 0.8

Results:
- l 0.4 outperforms all the models across all metrics, except for correlation (at 0.93 with other models around 0.97)
- l 0.8 and l trainable perform very strongly on correlation, MAE-OBS, LL OBS.
- l 0.2 is everytime the second-best model on every metrics

Comment:
- Keep 0.2, 0.4 and 0.8
- Test them on another window and see the results 

### Second experiment 
I run 
- spatial 2 RBF 0.2, 0.01, periodic l 0.2
- spatial 2 RBF 0.2, 0.01, periodic l 0.4
- spatial 2 RBF 0.2, 0.01, periodic l 0.8
on the first week of may 2025 (rather then the first week of november 2025).

Results: 
- periodic l0.8 outperforms the other models across all metrics 

### Final decision 
It seems reasonable to use as best model 
- spatial 2 RBF 0.2, 0.01, periodic l 0.8 





## Combine the spatial 2 RBF + GNN

### First experiment 
The idea is to have the GNN to forecast the number of ships. Then, the LGCP split them on the most likely cells. 
The gamma parameter controls the forecasting-concentration. With gamma = 1 is the same would be the standard LGCP. With gamma greater than 1 concentrates activity into higher-rate cells. 

I compared the following models
- spatial 2 RBF 0.2, 0.01, periodic l trianable (it reaches 1.6) 
- spatial 2 RBF 0.2, 0.01, periodic l 0.2
- spatial 2 RBF 0.2, 0.01, periodic l trianable + GNN gamma 1
- spatial 2 RBF 0.2, 0.01, periodic l trianable + GNN gamma 1.5
- spatial 2 RBF 0.2, 0.01, periodic l trianable + GNN gamma 2
- Pure GNN
- spatial 2 RBF 0.2, 0.01, periodic l 0.8 + GNN gamma 1

Results:

The strongest models are 
- spatial 2 RBF 0.2, 0.01, periodic l 0.8 + GNN gamma 1
- Pure GNN
- spatial 2 RBF 0.2, 0.01, periodic l trianable + GNN gamma 1
- spatial 2 RBF 0.2, 0.01, periodic l trianable + GNN gamma 1.5
They outperform all the other models across all metrics, except for MAE-OBS (0.32/0.31 vs 0.30 of the standard 2 spatial RBF)
The best overall is 
- spatial 2 RBF 0.2, 0.01, periodic l 0.8 + GNN gamma 1

In terms of MAE-OBS the best is 
- spatial 2 RBF 0.2, 0.01, periodic l trianable (it reaches 1.6) 
However, we saw that for this purpose, the classifier is even better. 

### Second experiment 
I compared 
- GNN
- LGCP spatial 2 RBF 0.2, 0.01, periodic l 0.8 
- LGCP+GNN gamma = 1.0
- LGCP+GNN gamma = 1.0, periodic l = 0.8
- LGCP+GNN gamma = 1.5
- LGCP+GNN gamma = 2.0
- LGCP+GNN+Classifier gamma = 1.0, periodic l 0.8

Results:
The best model is 
- LGCP+GNN+Classifier gamma = 1.0, periodic l 0.8
It outperforms all the other models across all metrics, except for OBS-LL.

Other two good models are 
- GNN
- LGCP+GNN gamma = 1.0, periodic l = 0.8
GNN is strong daily and outperforms other models on OBS-LL.
"LGCP+GNN gamma = 1.0, periodic l = 0.8" outperforms other models on OBS-RMSE, and it is good across all metrics.

A more interpretable model is
- LGCP spatial 2 RBF 0.2, 0.01, periodic l 0.8 
which is the second-best on correlation and MAE-OBS, weak on daily and decent for the remaining metrics. 

Conclusions:
- LGCP+GNN+Classfier is the best model found so far 
- LGCP+GNN seems to be better than LGCP
- fixing temporal periodic lengthscale at 0.8 is the right choice
- gamma seems optimal at 1, but could be okay also in 1.5

# Classifier tuning 

The main classifier tuning has been done one the 24-30 of may and 24-30 of november 2025. 

## Threshold 
I report the results on 1-7 november 2025, in a previous configuration:

0.2
daily_delta_corr: 0.93225868270584
daily_direction_accuracy_moving: 1.0
mae_daily: 7.978039868175983
mae_obs: 0.24169953591013787
mean_ll_daily: -5.2922074009115585
mean_ll_obs: -1.4498644479353768
rmse_daily: 9.711791701091496
rmse_obs: 0.6694534401518167

0.3
daily_delta_corr: 0.8401705016296871
daily_direction_accuracy_moving: 1.0
mae_daily: 8.973428382671305
mae_obs: 0.23108842462279533
mean_ll_daily: -6.786799265355976
mean_ll_obs: -1.8355087599948503
rmse_daily: 10.899511021763644
rmse_obs: 0.674025807591469

0.1
daily_delta_corr: 0.9517232064127957
daily_direction_accuracy_moving: 1.0
mae_daily: 6.492092413295593
mae_obs: 0.25801832260938673
mean_ll_daily: -3.858063230922393
mean_ll_obs: -1.0953725102815137
rmse_daily: 8.0052214212521
rmse_obs: 0.6704590115862532

Results:
- on 03-09 november and 05-11 may 0.2 is worse than 0.3 on the main metrics and rarely it is slightly slightly better

Conclusion:
- i will keep hard classifier threshold as 0.3 

## Layer 
I tested for the hard classfier, the layers 
- 16
- 32
- 64
- [16,32]
To choose which one is the best I set as criteria the correlation, mae-obs, rmse-obs, wesserstain and in case of similar results the other metrics, excluded the daily ones. 

Results
- On may 16 and 64 are the strongest 
- On november the results are very similar with 16 slighlty better

Conclusion
- I chose 16 as layer 

## Activation
I tried relu vs tanh. 

Results:
- almost equal 
- super super slightly better relu, but almost equal 

Conclusion
- I chose relu 

## Alpha

I tested the following orders of magnitude for alpha:

- 1e-2
- 1e-3
- 1e-4
- 1e-5

I prioritized daily correlation, observation-level MAE and RMSE, and the
Wasserstein distance.

Results:

- **May:** All four values produced very similar results on the primary
  metrics. On the secondary metrics, `1e-4` and `1e-5` generally performed
  best.
- **November:** `1e-4` performed best across the primary metrics. `1e-2`
  was also competitive, whereas `1e-5` was among the worst-performing
  values for most metrics.

Conclusion:
- I selected **alpha = 1e-4**.

## Max number of iter 

I tested the max number of iter 
- 500
- 1000

Results:
- the results are the same 

Conclusion:
- I chose 500 for simplicity 

## Threshold 

Given the current configuration, I tested six thresholds: 0.1, 0.2, 0.3,
0.4, 0.5, and 0.6. The experiments were evaluated on 24-30 May 2025 and
24-30 November 2025.

The primary selection metrics are daily correlation (`daily_delta_corr`),
MAE-OBS, RMSE-OBS, and Wasserstein distance. LL-OBS, accuracy, precision, and
recall are secondary metrics.

Results: 24-30 May 2025

| Threshold | Correlation (higher is better) | MAE-OBS | RMSE-OBS | Wasserstein |
|---:|---:|---:|---:|---:|
| 0.1 | **0.3993** | 0.3878 | **1.1885** | 0.2475 |
| 0.2 | 0.3002 | 0.3798 | 1.1971 | **0.2430** |
| 0.3 | 0.3062 | **0.3777** | 1.1978 | 0.2441 |
| 0.4 | 0.2811 | 0.3791 | 1.2041 | 0.2475 |
| 0.5 | 0.2149 | 0.3898 | 1.2335 | 0.2644 |
| 0.6 | 0.1172 | 0.3965 | 1.2517 | 0.2895 |

| Threshold | LL-OBS (higher is better) | Accuracy | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.1 | **-0.8413** | 0.7901 | 0.4769 | **0.9394** |
| 0.2 | -1.6599 | 0.8746 | 0.6533 | 0.7424 |
| 0.3 | -1.8190 | 0.8921 | 0.7302 | 0.6970 |
| 0.4 | -2.1075 | **0.8980** | 0.7719 | 0.6667 |
| 0.5 | -2.9324 | 0.8892 | 0.7917 | 0.5758 |
| 0.6 | -3.7415 | 0.8805 | **0.8378** | 0.4697 |

Results: 24-30 November 2025

| Threshold | Correlation (higher is better) | MAE-OBS | RMSE-OBS | Wasserstein |
|---:|---:|---:|---:|---:|
| 0.1 | 0.7881 | 0.4046 | **1.5147** | 0.2371 |
| 0.2 | **0.8366** | **0.3824** | 1.5362 | **0.2019** |
| 0.3 | 0.8330 | 0.3843 | 1.5428 | 0.2045 |
| 0.4 | 0.8163 | 0.3886 | 1.5907 | 0.2199 |
| 0.5 | 0.7940 | 0.4005 | 1.6178 | 0.2475 |
| 0.6 | 0.7559 | 0.4143 | 1.6356 | 0.2651 |

| Threshold | LL-OBS (higher is better) | Accuracy | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.1 | **-1.4243** | 0.7784 | 0.3168 | **0.8205** |
| 0.2 | -2.7933 | 0.9125 | 0.6047 | 0.6667 |
| 0.3 | -3.2524 | **0.9155** | **0.6389** | 0.5897 |
| 0.4 | -4.7576 | 0.9096 | 0.6250 | 0.5128 |
| 0.5 | -5.6451 | 0.9038 | 0.6364 | 0.3590 |
| 0.6 | -6.2972 | 0.8950 | 0.5882 | 0.2564 |

Comparison:

An equal-weight rank over the four primary metrics gives the following result
(a lower rank sum is better):

| Threshold | May rank sum | November rank sum | Combined rank sum |
|---:|---:|---:|---:|
| 0.1 | 9 | 15 | 24 |
| **0.2** | 9 | **5** | **14** |
| 0.3 | **8** | 9 | 17 |
| 0.4 | 14 | 13 | 27 |
| 0.5 | 20 | 18 | 38 |
| 0.6 | 24 | 24 | 48 |

The results show a clear deterioration for thresholds of 0.4 and above.
Threshold 0.1 gives the best RMSE-OBS in both windows, as well as the best
LL-OBS and recall. However, it has substantially lower accuracy and precision.
Thresholds 0.2 and 0.3 provide the strongest balance. In particular, 0.2 is
best on three of the four primary metrics in November and remains competitive
in May. Threshold 0.3 has the best MAE-OBS in May and slightly better
classification accuracy and precision, but is less consistent across the two
windows.

LL-OBS decreases rapidly as the threshold increases because hard gating can
set the predicted intensity to zero for an observed positive cell. Such false
negatives receive a very large likelihood penalty. This also explains why the
low threshold and its high recall perform particularly well on LL-OBS.

Conclusion:

- I select **threshold = 0.2** because it is the most robust choice across the
  two windows according to the four primary metrics.
- Threshold 0.1 would be preferable if recall, LL-OBS, or the cost of missing
  active cells became the main concern. The price is on accuracy and precision. 
- Threshold 0.3 would be preferable only if precision and false-positive
  reduction were given more importance.
- Thresholds greater than or equal to 0.4 are discarded.
- A possible final refinement is to test 0.15, 0.175, 0.20, and 0.225 on
  additional validation windows. The optimum is likely to lie in this range.

## Confidence-weighted hard redistribution

The `confidence_redistribute` gate redistributes only a confidence-dependent
part of the intensity removed by the hard classifier. It assigns this mass to
accepted cells according to their LGCP intensity and classifier confidence.
The first tests (`gamma = 1`, `scale = 1`) show that it is a useful compromise
between the hard classifier and full hard redistribution.

I compared the tuned classifier (`threshold = 0.2`, layer `[16]`) with the
original classifier settings (`threshold = 0.3`, layer `[32]`) on May 5--11 and
November 3--9. Excluding LL-OBS, the original settings were better on three of
the four primary metrics in both windows and consistently improved accuracy
and precision. Therefore, the remaining confidence-redistribution experiments
use **threshold 0.3 and layer `[32]`**.

