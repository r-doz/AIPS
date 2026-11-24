**SVGP residual variance**

We thank the reviewer for this suggestion. We added the diagonal residual variance r_i = k(x_i, x_i) − [A K_uf]_ii to the latent field, f_i = a_iᵀu + √r_i ε_i, both in the ELBO and in the predictive distribution. Since K_fu and A are already computed, r_i costs one row-wise sum, so the cost is unchanged (median training time 14.0 vs 14.1 min per run). We retrained the model with the same configurations, seeds, classifier and gate, and evaluated it with the protocol of the paper (8 test weeks × 10 seeds).

| Model | Month | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|
| Red-LGCP (paper) | May | −0.4413 | **0.2075** |
| | Nov. | −0.4553 | **0.2085** |
| Red-LGCP + residual variance | May | **−0.4398** | 0.2093 |
| | Nov. | **−0.4519** | 0.2089 |

**CRPS and log-likelihood.** The residual variance slightly improves the log-likelihood (better in 6 of the 8 test weeks) and slightly worsens the CRPS (worse in 7 of 8). Neither difference is significant (paired Wilcoxon, p = 0.08 and 0.06). The point metrics change by at most 0.01, except RMSE in May (0.941 → 0.966), and none of these changes is significant either.

**GP contribution.** This is where the residual variance matters. Without it, the variance that the 300 inducing points cannot represent (84–86% of the prior variance, mostly from the short-lengthscale kernel components) is dropped from the ELBO, so the kernel variances can grow at no cost. With it, the model shrinks the prior variance of the GP about 12-fold, and the total latent variance at the test points ends up close to before, which is why the scores barely change. The posterior mean of the GP then explains about a third of the variance it did before, but its structure is the same: the static component dominates, followed by a 14-day cycle and a negligible short-term part. The covariate effects change by less than 0.01, except the weekend and holiday effects, which become slightly weaker (weekend: 0.74 → 0.81 in May).

| May / Nov. | Red-LGCP (paper) | + residual variance |
|---|---|---|
| Prior variance of the GP, k(x, x) | 1.03 / 0.89 | 0.09 / 0.07 |
| Variance of the GP posterior mean | 0.14 / 0.15 | 0.05 / 0.05 |
| Latent variance at the test points | 0.077 / 0.056 | 0.082 / 0.066 |
| Static / periodic / short-term share (May) | 0.80 / 0.16 / 0.01 | 0.83 / 0.15 / 0.00 |
| Learned period (days, May) | 13.96 | 14.02 |

**Changes to the paper.** We will add this experiment to the appendix and note, where we introduce the SoR approximation, that the residual variance can be restored at no extra cost: it leaves the predictive scores and the interpretation of the latent field (Table 6) essentially unchanged, but gives a smaller GP component. We will also add a caveat on the kernel variances: under SoR, only 14–16% of the prior variance k(x, x) reaches the likelihood, so the learned kernel variances overstate the effective prior variance of the GP and should not be read at face value. The relative shares of the latent field and the covariate effects, which are what we interpret, do not depend on this choice.
