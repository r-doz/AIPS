**Response to Reviewer 1**

We thank the reviewer for these suggestions, which we have implemented in full. All experiments follow the protocol of the paper: the same 8 test weeks, every new choice (thresholds, hyperparameters) made on the 4 validation weeks only, 10 seeds for the neural models, and paired two-sided Wilcoxon tests over the test weeks with Holm correction. Red-LGCP is unchanged. In all tables, values are means over the 4 test weeks of each month (× 10 seeds for the neural models), and bold marks the best value per column and month.

**1. Validation-tuned threshold for the baselines.** Each Poisson baseline now predicts a cell as active when 1 − exp(−λ̂) ≥ τ, instead of λ̂ > 0. For every baseline, τ was chosen on the validation weeks, separately for accuracy and for F1, which is the most favourable setting for the baselines.

| Model | Month | Accuracy (λ̂ > 0) | Accuracy (tuned τ) | F1 (λ̂ > 0) | F1 (tuned τ) |
|---|---|---|---|---|---|
| LAP | May | 0.885 | 0.882 | 0.543 | 0.543 |
| | Nov. | 0.875 | 0.888 | 0.490 | 0.490 |
| GCMP | May | 0.254 | 0.869 | 0.259 | 0.468 |
| | Nov. | 0.212 | 0.878 | 0.235 | 0.447 |
| Poisson GLM | May | 0.131 | 0.903 | 0.231 | 0.580 |
| | Nov. | 0.122 | **0.904** | 0.217 | 0.591 |
| GNN | May | 0.131 | 0.901 | 0.231 | 0.572 |
| | Nov. | 0.122 | 0.893 | 0.217 | 0.531 |
| ConvLSTM | May | 0.131 | 0.904 | 0.231 | 0.580 |
| | Nov. | 0.122 | 0.902 | 0.217 | **0.592** |
| **Red-LGCP (ours)** | May | **0.912** | **0.912** | **0.645** | **0.645** |
| | Nov. | **0.899** | 0.899 | **0.573** | 0.573 |

Red-LGCP uses its own gate in both columns. With tuned thresholds, the accuracy of GCMP, GLM, GNN and ConvLSTM rises from 0.12–0.25 to 0.87–0.90. Red-LGCP remains the best in May, but in November the GLM and ConvLSTM are slightly better, and its accuracy and F1 advantages over the GLM, GNN and ConvLSTM are no longer significant.

**2. Hurdle-Poisson GLM.** A logistic regression for P(Y > 0) and a zero-truncated Poisson GLM for the positive counts, with the inputs of the Poisson GLM. The window length and both L2 penalties were selected on the validation weeks with the criterion of App. B.2. Its point forecasts use the gate and redistribution of Red-LGCP with a validation-tuned threshold.

| Model | Month | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|---|
| Poisson GLM (tuned τ) | May | 0.330 | 1.377 | 0.222 | 0.903 | 0.580 | −0.617 | 0.277 |
| | Nov. | 0.311 | 1.197 | 0.229 | **0.904** | 0.591 | −0.643 | 0.267 |
| Hurdle-Poisson GLM | May | 0.299 | 0.953 | 0.107 | 0.910 | 0.633 | −0.476 | 0.217 |
| | Nov. | 0.308 | 0.976 | 0.107 | 0.903 | **0.607** | −0.468 | 0.226 |
| **Red-LGCP (ours)** | May | **0.286** | **0.941** | **0.101** | **0.912** | **0.645** | **−0.441** | **0.208** |
| | Nov. | **0.283** | **0.881** | **0.085** | 0.899 | 0.573 | **−0.455** | **0.209** |

The hurdle structure alone improves the GLM substantially (e.g., log-likelihood from −0.617 to −0.476 in May). Red-LGCP is better than the hurdle GLM on every metric except accuracy and F1 in November, but none of these differences is significant.

**3. Neural baselines with lag t−7 and a hurdle head.** The GNN already used the counts at t−1 and t−7, so we added a hurdle head (activity probability and zero-truncated Poisson rate) trained with the hurdle likelihood. For the ConvLSTM, we added the counts at t−1 and t−7 as input channels and trained it both with its Poisson head and with a hurdle head. Hyperparameters were re-selected on the validation weeks as in App. B.2, and the hurdle models use the gate and redistribution of Red-LGCP with a validation-tuned threshold.

| Model | Month | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|---|
| GNN (tuned τ) | May | 0.316 | 0.948 | 0.165 | 0.901 | 0.572 | −0.521 | 0.232 |
| | Nov. | 0.307 | 0.912 | 0.183 | 0.893 | 0.531 | −0.589 | 0.232 |
| GNN + hurdle | May | 0.318 | 0.942 | 0.142 | 0.902 | 0.581 | −0.478 | 0.224 |
| | Nov. | 0.315 | 0.920 | 0.158 | 0.888 | 0.520 | −0.482 | 0.227 |
| ConvLSTM (tuned τ) | May | 0.304 | 0.945 | 0.182 | 0.904 | 0.580 | −0.512 | 0.226 |
| | Nov. | 0.289 | 0.844 | 0.156 | **0.902** | 0.592 | −0.524 | 0.212 |
| ConvLSTM + lags | May | 0.300 | **0.899** | 0.130 | 0.890 | 0.576 | −0.488 | 0.216 |
| | Nov. | 0.288 | 0.838 | 0.128 | 0.891 | 0.592 | −0.516 | 0.210 |
| ConvLSTM + lags + hurdle | May | 0.308 | 0.908 | 0.131 | 0.901 | 0.566 | −0.456 | 0.214 |
| | Nov. | 0.286 | **0.832** | 0.110 | 0.896 | **0.599** | **−0.450** | **0.202** |
| **Red-LGCP (ours)** | May | **0.286** | 0.941 | **0.101** | **0.912** | **0.645** | **−0.441** | **0.208** |
| | Nov. | **0.283** | 0.881 | **0.085** | 0.899 | 0.573 | −0.455 | 0.209 |

The hurdle head and the extra lags improve both networks. Red-LGCP is significantly better than the GNN with a hurdle head in F1, Wasserstein distance and log-likelihood. The ConvLSTM with lags and a hurdle head is the strongest baseline: it has a lower RMSE and, in November, a better F1, log-likelihood and CRPS, while Red-LGCP keeps a lower MAE and a significantly lower Wasserstein distance.

In the tables of points 2 and 3, rows marked "tuned τ" use the thresholds of point 1 and set the cells below the threshold to zero; accuracy uses the accuracy-optimal threshold and the other metrics the F1-optimal one.

**Summary and changes to the paper.** The fairer baselines narrow the gap considerably. Red-LGCP keeps the best MAE in both months and a significantly lower Wasserstein distance than every Poisson and neural baseline, and adding the hurdle structure improves the log-likelihood and CRPS of every baseline, consistent with our ablation. We will add these baselines and results to Sec. 5 and the appendix, and revise our claims accordingly: Red-LGCP matches the strongest baselines in point and probabilistic accuracy, gives the best distributional fit, and, unlike them, provides interpretable covariate effects and a latent field.
