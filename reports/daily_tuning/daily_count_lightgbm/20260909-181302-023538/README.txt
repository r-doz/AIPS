Ranking: equal-weight mean window MAE; tie-break by mean window RMSE.
Trial 0 is the original configuration. Each window trains on all earlier days.
All eight windows are used for model selection, not independent test evaluation.
best_config.yaml preserves the base forecast dates: change these to an untouched
test window before reporting final performance. The tuner does not refit/deploy a model.
