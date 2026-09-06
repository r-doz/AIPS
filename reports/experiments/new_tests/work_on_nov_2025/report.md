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

## Tune the singol spatial RBF again
After discovering the best 2 spatial RBF is 
- lgcp_spat_2rbf_l0.20_l0.01
I tried the singol spatial RBF with 
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
- The model "temporal periodic RBF with lengthscale NON-trainbale" is better or equal to "2 RBF spatial 0.2, 0.01 and temporal 0.01" on all metrics, but MAE-OBS (0.30 vs 0.32). Moreover, the plot shows more peak predictions.
- The lr 0.005 makes a worse results on all the metrics 

Comment:
The temporal periodic lengthscale controls how similar two times within a cycle must be. In other words, if l is high two days within the week (the period) should be similar, while with l low, two days within the week could be really different. 


## Combine the spatial 2 RBF + GNN
The idea is to have the GNN to forecast the number of ships. Then, the LGCP split them on the most likely cells. 
The gamma parameter controls the forecasting-concentration. With gamma = 1 is the same would be the standard LGCP. With gamma greater than 1 concentrates activity into higher-rate cells. 










