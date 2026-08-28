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

## New tests to do 
Since decreasing the threshold means improvement for the classifier, another test could be done by decreasing the trheshold from 0.03.  

Another class of test regards the space intensity. One visual problem is that our model is prudent, the forcast tend to be wide but with small values. So, 3 tests have to be done:
- same setting, without classifier, with space lengscale 0.1, variance = 4, false, false
- add a further spatial RBF, so: RBF(0.2, 2, F, T) + RBF(0.03, 0.5, F, T)
- add a further spatial RQ: so, RBF(0.1, 2, F, T) + RQ(0.05, 0.5, 1, F, T, T) 


