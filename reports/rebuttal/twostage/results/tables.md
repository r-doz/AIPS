## Checks

- S0 log-lik. recomputed vs Table 5 (max abs gap): 4.27e-05
- S0 CRPS recomputed vs Table 5 (max abs gap): 1.34e-05
- S0 MAE recomputed with the retrained gate classifier vs saved (max abs gap): 0.00176
- S0 MAE recomputed vs saved (mean abs gap): 0.000426
- S0 gate decisions reproduced (min share of cell-days per run): 1
- S0 gate decisions reproduced (mean share): 1
- rows with extended pmf support (total): 0

## Ladder (mean ± population std over 4 weeks × 10 seeds)

| Step | Month | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|---|
| Two-stage (paper, Table 5) | May | 0.304 ± 0.079 | 1.244 ± 0.482 | 0.164 ± 0.075 | 0.895 ± 0.013 | 0.632 ± 0.068 | -0.520 ± 0.082 | 0.247 ± 0.068 |
| Two-stage (paper, Table 5) | Nov | 0.286 ± 0.061 | 0.945 ± 0.374 | 0.137 ± 0.029 | 0.891 ± 0.020 | 0.577 ± 0.084 | -0.525 ± 0.054 | 0.226 ± 0.044 |
| + hurdle scoring | May | 0.304 ± 0.079 | 1.244 ± 0.482 | 0.164 ± 0.075 | 0.895 ± 0.013 | 0.632 ± 0.068 | -0.498 ± 0.084 | 0.240 ± 0.070 |
| + hurdle scoring | Nov | 0.286 ± 0.061 | 0.945 ± 0.374 | 0.137 ± 0.029 | 0.891 ± 0.020 | 0.577 ± 0.084 | -0.520 ± 0.053 | 0.226 ± 0.050 |
| + MC-dropout classifier, gate τ + σ | May | 0.304 ± 0.075 | 1.272 ± 0.476 | 0.162 ± 0.083 | 0.912 ± 0.009 | 0.641 ± 0.071 | -0.498 ± 0.088 | 0.242 ± 0.074 |
| + MC-dropout classifier, gate τ + σ | Nov | 0.282 ± 0.064 | 0.950 ± 0.375 | 0.132 ± 0.030 | 0.900 ± 0.021 | 0.579 ± 0.098 | -0.526 ± 0.061 | 0.225 ± 0.052 |
| + zero-truncated LGCP (Red-LGCP trained in two stages) | May | 0.288 ± 0.040 | 0.957 ± 0.114 | 0.101 ± 0.047 | 0.912 ± 0.009 | 0.641 ± 0.071 | -0.441 ± 0.060 | 0.208 ± 0.041 |
| + zero-truncated LGCP (Red-LGCP trained in two stages) | Nov | 0.281 ± 0.057 | 0.879 ± 0.269 | 0.087 ± 0.014 | 0.900 ± 0.021 | 0.579 ± 0.098 | -0.454 ± 0.041 | 0.209 ± 0.043 |
| Red-LGCP, joint (paper) | May | 0.286 ± 0.041 | 0.941 ± 0.115 | 0.101 ± 0.047 | 0.912 ± 0.009 | 0.645 ± 0.072 | -0.441 ± 0.061 | 0.208 ± 0.042 |
| Red-LGCP, joint (paper) | Nov | 0.283 ± 0.058 | 0.881 ± 0.278 | 0.085 ± 0.015 | 0.899 ± 0.021 | 0.573 ± 0.096 | -0.455 ± 0.042 | 0.209 ± 0.043 |

## Paired tests (weeks out of 8 where the first is better; exact Wilcoxon p, uncorrected)

| Comparison | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|
| S1 vs S0 | identical | identical | identical | identical | identical | 6/8 (p=0.078) | 6/8 (p=0.383) |
| S2 vs S1 | 5/8 (p=0.641) | 2/8 (p=0.078) | 6/8 (p=0.148) | 7/8 (p=0.016) | 5/8 (p=0.844) | 3/8 (p=0.461) | 4/8 (p=1.000) |
| S3 vs S2 | 3/8 (p=0.945) | 5/8 (p=0.148) | 8/8 (p=0.008) | identical | identical | 8/8 (p=0.008) | 7/8 (p=0.023) |
| S3 vs joint | 2/8 (p=0.461) | 2/8 (p=0.195) | 3/8 (p=0.250) | 5/8 (p=0.578) | 4/8 (p=0.742) | 5/8 (p=0.312) | 2/8 (p=0.195) |
| S0 vs joint | 3/8 (p=0.742) | 3/8 (p=0.312) | 0/8 (p=0.008) | 1/8 (p=0.023) | 3/8 (p=0.742) | 0/8 (p=0.008) | 0/8 (p=0.008) |

## Share of the gap between the paper's two-stage model and joint Red-LGCP closed by each step

| Step | Wass. ↓ | Acc. ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|
| S1 | -0% | +0% | +18% | +11% |
| S2 | +6% | +103% | -4% | -1% |
| S3 | +92% | +0% | +87% | +89% |
| remaining (joint - S3) | +2% | -3% | -1% | +2% |

Mean difference joint − two-stage (the gap): Wass. ↓ -0.0575, Acc. ↑ +0.0126, Log-lik. ↑ +0.0743, CRPS ↓ -0.0282
