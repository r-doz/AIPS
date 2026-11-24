**Two-stage ablation**

We thank the reviewer; the reviewer is right. In Red-LGCP the classifier and the count model share no parameters, and the ELBO is the sum of a classifier term and a count term (plus the KL of the GP), so joint training exchanges no gradients between the two parts. Our two-stage ablation therefore did not isolate joint training: it differed from Red-LGCP in exactly the three ways the reviewer lists. We added the three changes one at a time, with the same configurations, seeds (10) and test weeks (8).

| Model | Month | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|---|
| Two-stage (Table 5) | May | 0.304 | 1.244 | 0.164 | 0.895 | 0.632 | −0.520 | 0.247 |
| | Nov. | 0.286 | 0.945 | 0.137 | 0.891 | 0.577 | −0.525 | 0.226 |
| + hurdle probability in LL/CRPS | May | 0.304 | 1.244 | 0.164 | 0.895 | 0.632 | −0.498 | 0.240 |
| | Nov. | 0.286 | 0.945 | 0.137 | 0.891 | 0.577 | −0.520 | 0.226 |
| + MC-dropout classifier, gate τ + σ | May | 0.304 | 1.272 | 0.162 | **0.912** | 0.641 | −0.498 | 0.242 |
| | Nov. | 0.282 | 0.950 | 0.132 | **0.900** | **0.579** | −0.526 | 0.225 |
| + zero-truncated LGCP on positive counts | May | 0.288 | 0.957 | **0.101** | **0.912** | 0.641 | **−0.441** | **0.208** |
| | Nov. | **0.281** | **0.879** | 0.087 | **0.900** | **0.579** | **−0.454** | **0.209** |
| Red-LGCP (joint training) | May | **0.286** | **0.941** | **0.101** | **0.912** | **0.645** | **−0.441** | **0.208** |
| | Nov. | 0.283 | 0.881 | **0.085** | 0.899 | 0.573 | −0.455 | **0.209** |

1. Scoring the two-stage model with its hurdle probability improves its log-likelihood and CRPS slightly (averaged over both months, −0.523 → −0.509 and 0.236 → 0.233; not significant).
2. Red-LGCP's MC-dropout classifier with the τ + σ gate closes the accuracy gap (0.893 → 0.906; better in 7 of 8 weeks, p = 0.02).
3. Replacing the Poisson LGCP fitted to all cell-days with the zero-truncated LGCP fitted to the positive counts closes about 90% of the gap to Red-LGCP in Wasserstein distance, log-likelihood and CRPS (better in 7–8 of 8 weeks, p ≤ 0.023).

The fourth row is Red-LGCP with its two parts trained separately. It is statistically indistinguishable from the jointly trained model on every metric (paired Wilcoxon over the 8 test weeks, all p ≥ 0.19).

**Changes to the paper.** We will replace the two-stage row of Table 5 with this decomposition and correct the text: Red-LGCP's gains over the two-stage model come from the zero-truncated count model and the uncertainty-aware gate, not from joint training, which is equivalent to training the two parts separately. We keep the joint formulation because it gives a single objective and a single predictive distribution.
