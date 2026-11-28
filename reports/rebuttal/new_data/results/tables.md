Runs per dataset and model: {('paper', 'Red-LGCP (paper data)'): 80, ('trieste_4y', 'GCMP'): 8, ('trieste_4y', 'GLM'): 8, ('trieste_4y', 'Hurdle GLM'): 8, ('trieste_4y', 'LAP'): 8, ('trieste_4y', 'Red-LGCP'): 80, ('venice', 'GCMP'): 8, ('venice', 'GLM'): 8, ('venice', 'Hurdle GLM'): 8, ('venice', 'LAP'): 8, ('venice', 'Red-LGCP'): 80}

## Venice–Chioggia, 2024–2025

Mean over 4 test weeks × 10 seeds (Red-LGCP) or 4 test weeks (deterministic baselines); baselines with validation-tuned thresholds (as in the reviewer-1 tables); best in bold.

| Model | Month | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|---|
| Red-LGCP | May | 0.953 | 1.725 | 0.536 | 0.772 | 0.719 | **-1.131** | 0.622 |
| LAP | May | 1.208 | 2.281 | **0.076** | 0.701 | 0.548 | -9.021 | 0.982 |
| GCMP | May | 1.188 | 1.888 | 0.828 | 0.650 | 0.576 | -1.639 | 0.844 |
| GLM | May | **0.863** | **1.699** | 0.459 | 0.788 | 0.704 | -1.331 | 0.675 |
| Hurdle GLM | May | 0.896 | 1.710 | 0.406 | **0.805** | **0.727** | -1.152 | **0.609** |
| Red-LGCP | Nov | **1.383** | **2.978** | 0.847 | 0.742 | **0.692** | **-1.385** | **0.969** |
| LAP | Nov | 1.755 | 3.827 | **0.008** | 0.689 | 0.515 | -13.112 | 1.492 |
| GCMP | Nov | 1.664 | 3.311 | 1.241 | 0.666 | 0.512 | -2.421 | 1.268 |
| GLM | Nov | 1.453 | 3.480 | 0.830 | 0.733 | 0.670 | -1.945 | 1.163 |
| Hurdle GLM | Nov | 1.418 | 3.020 | 0.671 | **0.762** | **0.692** | -1.435 | 0.989 |

Red-LGCP vs each baseline: weeks out of 8 where Red-LGCP is better; exact Wilcoxon p, Holm-corrected over the 4 baselines

| vs | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|
| LAP | 8/8 (p=0.031) | 8/8 (p=0.031) | 0/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) |
| GCMP | 8/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) |
| GLM | 2/8 (p=0.766) | 4/8 (p=0.922) | 3/8 (p=0.109) | 4/8 (p=0.742) | 8/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) |
| Hurdle GLM | 3/8 (p=0.844) | 4/8 (p=0.922) | 0/8 (p=0.031) | 2/8 (p=0.078) | 3/8 (p=0.641) | 6/8 (p=0.039) | 3/8 (p=0.945) |

## Gulf of Trieste, 2022–2025 (4-year history)

Mean over 4 test weeks × 10 seeds (Red-LGCP) or 4 test weeks (deterministic baselines); baselines with validation-tuned thresholds (as in the reviewer-1 tables); best in bold.

| Model | Month | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|---|
| Red-LGCP | May | **0.259** | 0.923 | 0.122 | **0.926** | **0.653** | **-0.396** | 0.196 |
| LAP | May | 0.321 | 1.024 | **0.041** | 0.908 | 0.539 | -2.272 | 0.254 |
| GCMP | May | 0.376 | 1.168 | 0.312 | 0.883 | 0.483 | -0.706 | 0.293 |
| GLM | May | 0.316 | 1.295 | 0.227 | 0.918 | 0.623 | -0.580 | 0.266 |
| Hurdle GLM | May | 0.273 | **0.878** | 0.107 | 0.920 | 0.599 | -0.417 | **0.194** |
| Red-LGCP | Nov | **0.283** | **0.974** | 0.096 | 0.903 | **0.593** | **-0.471** | **0.216** |
| LAP | Nov | 0.378 | 1.199 | **0.013** | 0.894 | 0.516 | -2.905 | 0.303 |
| GCMP | Nov | 0.353 | 1.055 | 0.272 | 0.875 | 0.467 | -0.695 | 0.267 |
| GLM | Nov | 0.375 | 1.802 | 0.280 | 0.897 | 0.516 | -0.672 | 0.300 |
| Hurdle GLM | Nov | 0.328 | 1.076 | 0.125 | **0.906** | 0.512 | -0.481 | 0.230 |

Red-LGCP vs each baseline: weeks out of 8 where Red-LGCP is better; exact Wilcoxon p, Holm-corrected over the 4 baselines

| vs | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|
| LAP | 8/8 (p=0.031) | 7/8 (p=0.062) | 0/8 (p=0.031) | 5/8 (p=0.445) | 8/8 (p=0.031) | 8/8 (p=0.031) | 8/8 (p=0.031) |
| GCMP | 8/8 (p=0.031) | 6/8 (p=0.156) | 8/8 (p=0.031) | 7/8 (p=0.062) | 7/8 (p=0.047) | 8/8 (p=0.031) | 8/8 (p=0.031) |
| GLM | 6/8 (p=0.047) | 6/8 (p=0.117) | 8/8 (p=0.031) | 5/8 (p=0.445) | 7/8 (p=0.156) | 8/8 (p=0.031) | 8/8 (p=0.031) |
| Hurdle GLM | 7/8 (p=0.047) | 4/8 (p=0.547) | 6/8 (p=0.461) | 4/8 (p=0.945) | 6/8 (p=0.156) | 6/8 (p=0.195) | 6/8 (p=0.250) |

## Gulf of Trieste: 4 years of history vs the paper's 2 years

| Model | Month | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|---|
| Red-LGCP, 4 years | May | **0.259** | **0.923** | 0.122 | **0.926** | **0.653** | **-0.396** | **0.196** |
| Red-LGCP, 2 years (paper) | May | 0.286 | 0.941 | **0.101** | 0.912 | 0.645 | -0.441 | 0.208 |
| Red-LGCP, 4 years | Nov | **0.283** | 0.974 | 0.096 | **0.903** | **0.593** | -0.471 | 0.216 |
| Red-LGCP, 2 years (paper) | Nov | **0.283** | **0.881** | **0.085** | 0.899 | 0.573 | **-0.455** | **0.209** |

4 years vs paper: weeks out of 8 where the 4-year model is better; exact Wilcoxon p

| vs | MAE ↓ | RMSE ↓ | Wass. ↓ | Acc. ↑ | F1 ↑ | Log-lik. ↑ | CRPS ↓ |
|---|---|---|---|---|---|---|---|
| Red-LGCP, 2 years (paper) | 7/8 (p=0.109) | 3/8 (p=0.383) | 1/8 (p=0.039) | 5/8 (p=0.195) | 6/8 (p=0.383) | 5/8 (p=0.312) | 4/8 (p=0.641) |

