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


## Classifier hard local redistribuite
I tried the version of the classifier in which if the classifier tells: "do not put vessels here" then the quantity is distribuited in the closest points rather then the original version in which they were re-distribuited across all the positions. 

Results:
- in may the local-redistribuite and the redistribuite have almost the same results 
- in november the local-redistribuite is worse then the redistribuite 

Conclusion: 
- I would discard the local-redistribuite version 