# Report 

## LGCP vs LGCP+GNN vs LGCP+GNN+Classifier 
I tried 5 models 
- GNN
- LGCP
- LGCP, temporal periodic l 0.8
- LGCP + GNN, periodic l 0.8, gamma = 1
- LGCP + GNN + Classifier, periodic l 0.8, gamma = 1

Results:
- GNN
is the best model. It outperforms all models across all metrics, except for MAE-OBS and RMSE-DAILY in which it is the second best. 
- LGCP + GNN, periodic l 0.8, gamma = 1
It is a good model, it is the best model on correlation, mae-daily, ll-daily, ll-obs and the second best model on rmse-daily and rmse-obs. 
- LGCP + GNN + Classifier, periodic l 0.8, gamma = 1
it is the best model on MAE-OBS by far. Moreover it is the first model with others in correlation. It is very bad on LL.
- LGCP periodic l 0.8
is a good model overall, everytime the best or second-best, except for correlation and mae-obs.

Conclusion:
- LGCP must have periodic l = 0.8
- LGCP+GNN+Classfier it is the best found so far from a OBS level. However it is weak on LL.
- It seems that LGCP+GNN simply inherit the daily good skills of GNN without outperforming it 


