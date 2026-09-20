# Final experiments 

## Description 
We do experimentation of our best models against some good models and baseline models on two 2025 windows on 3 different split. 

Out models are 
- LGCP 1 spatial kernel (lgcp_1_spatial_kernel)
- LGCP 2 spatial kernel (lgcp_2_spatial_kernels)
- LGCP + classifier hard (lgcp_2_spatial_kernels_classifier_hard)
- LGCP + classifier redistribuite (lgcp_2_spatial_kernels_classifier_redistribuite)
The comparison good models are 
- gnn 
- convLSTM
- window-GLM
The baselines are 
- last available Poisson - one step ahead version 
- global cell mean - expanding version 

The considered window are 
- 2025 may 05-11
- 2025 november 03-09

The different split are 
- 1 week 
- 3 days
- 1 days for 7 times + mean 
To be more specific, in the version 1 week we train the model, then we forecast the first day of the week $g_1$, then we provide $g_1$ to the model and we forcast $g_2$ and so on. 
With the 3 days forcasting the idea is the same. 
For 1 day forcasting, we train the model and we forcast $g_1$, then we train the model including also $g_1$ and we forcast $g_2$, then we train the model again with also using $g_1$ and $g_2$ and we forcast $g_3$ and so on until $g_7$. At the end, we average the metrics of the 7 tests (one for each day) and we return that average. 

## Period correction 
During the tuning we used 0.074 as fixed period for the periodic temporal kernel. Our aim was to fix it to have periods of 1 week. However, if the train data change in length then this number should have been adapted. Therefore, we changed the config in order to be able to select as an integer number the "days_period" and we fix it for all experiments as 7. The algorithm autonomously define the proper value depending on the length of the train data. 

Results with 7 days period:
On november: \
- the fixed kernel at 7 days seems for "2 spatiak kernels" btter on an obs level, better in tren, a bit worse in correlation, worse in daily predictions. 
- the classifier hard is better overall 
- the classifier redistribuite is better on an observation level, better in trend, and W measure, a bit worse in correlation and worse in daily predictions. 
On may:
- all models with the fixed period at 7 days are worse across all metrics. 

Results with no period:
- all models across almost all variables seem to be worse with no period rather than with 0-074 period 

Results with 14 days period:
14 days period is similar to 0.074 because it may 0.074 is 10-11 days while in november is almost 14 days. 

Results with 14 days period with period trainable
- it is very good on may, decent, almost as 0.074 for november 
- currently is our choice 


## Comparison

Model results on the full test weeks, using the same tuned hyperparameters
used throughout the paper.

### May 5--11, 2025

| Method | LL_obs | MAE_obs | RMSE_obs | Wasserstein | Corr_delta | Accuracy | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last Available Poisson | -3.390 | 0.388 | 1.136 | **0.032** | 0.049 | 0.872 | 0.545 | 0.612 |
| Global Cell Mean Poisson | -0.634 | 0.396 | 0.995 | 0.322 | -0.667 | 0.283 | **1.000** | 0.183 |
| Window Poisson GLM | -0.730 | 0.420 | 1.005 | 0.388 | 0.648 | 0.160 | **1.000** | 0.160 |
| GNN | -0.637 | 0.377 | 1.003 | 0.226 | 0.910 | 0.160 | **1.000** | 0.160 |
| ConvLSTM | **-0.572** | 0.352 | **0.879** | 0.212 | 0.619 | 0.160 | **1.000** | 0.160 |
| LGCP | -0.589 | 0.371 | 0.973 | 0.255 | 0.796 | 0.160 | **1.000** | 0.160 |
| LGCP + Classifier H | -1.733 | **0.307** | 0.977 | 0.188 | **0.925** | **0.892** | 0.764 | **0.636** |
| LGCP + Classifier HR | -1.144 | 0.336 | 1.037 | 0.136 | 0.796 | 0.641 | 0.873 | 0.293 |
| LGCP + Classifier HLR | -1.149 | 0.337 | 1.039 | 0.136 | 0.796 | 0.641 | 0.873 | 0.293 |

### November 3--9, 2025

| Method | LL_obs | MAE_obs | RMSE_obs | Wasserstein | Corr_delta | Accuracy | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last Available Poisson | -2.473 | 0.318 | 0.788 | **0.009** | 0.295 | 0.866 | 0.547 | 0.569 |
| Global Cell Mean Poisson | -0.791 | 0.433 | 0.876 | 0.269 | -0.243 | 0.257 | **1.000** | 0.172 |
| Window Poisson GLM | -0.613 | 0.405 | 0.764 | 0.372 | 0.897 | 0.155 | **1.000** | 0.155 |
| GNN | -0.552 | 0.346 | 0.736 | 0.168 | 0.884 | 0.155 | **1.000** | 0.155 |
| ConvLSTM | **-0.509** | 0.255 | **0.587** | 0.139 | 0.657 | 0.155 | **1.000** | 0.155 |
| LGCP | -0.570 | 0.334 | 0.699 | 0.242 | **0.938** | 0.155 | **1.000** | 0.155 |
| LGCP + Classifier H | -1.797 | 0.250 | 0.683 | 0.182 | 0.824 | **0.892** | 0.660 | **0.648** |
| LGCP + Classifier HR | -1.613 | **0.239** | 0.604 | 0.115 | **0.938** | 0.618 | 0.698 | 0.243 |
| LGCP + Classifier HLR | -1.639 | 0.270 | 0.710 | 0.106 | **0.938** | 0.618 | 0.698 | 0.243 |

**Bold** indicates the best value within each week. Lower values are better
for MAE_obs, RMSE_obs, and Wasserstein; higher values are better for all other
metrics. `Corr_delta` is the correlation of daily changes.

Classifier abbreviations: H = hard, HR = hard redistribution, and HLR = hard
local redistribution.
