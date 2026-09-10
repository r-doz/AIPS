# Report 

## LGCP vs LGCP+GNN vs LGCP+GNN+Classifier 
I tried 5 models 
- GNN
- LGCP
- LGCP, temporal periodic l 0.8
- LGCP + GNN, periodic l 0.8, gamma = 1
- LGCP + GNN + Classifier, periodic l 0.8, gamma = 1

Results:
- LGCP + GNN, periodic l 0.8, gamma = 1
is the best model on mae-daily, ll-daily, rmse-daily and the second-best on ll-obs, rmse-obs.
- GNN
is the best model on mae-daily, ll-daily, ll-obs, rmse-daily and the second-best model on mae-obs
- LGCP + GNN + Classifier, periodic l 0.8, gamma = 1
is by far the best model on mae-obs (0.23 vs 0.33) and on rmse-obs, and the second-best on mae-daily and rmse-daily
- LGCP, temporal periodic l 0.8
ramins a good and interpretable model overall. Especially, it is the best on correlation (0.95 vs 0.70).

Conclusion:
- LGCP must have periodic l = 0.8
- LGCP+GNN+Classfier it is the best found so far from a OBS level. However it is weak on LL.
- It seems that LGCP+GNN simply inherit the daily good skills of GNN without outperforming it 

## Catboos vs GNN as a support for the LGCP
I tried Catboost as a daily predictor instead of GNN. 

Result:
- LGCP+GNN outperforms LGCP+Catboost across all metrics


