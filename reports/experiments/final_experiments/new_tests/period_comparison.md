# Temporal period comparison

May: 5–11 May 2025. November: 3–9 November 2025. All models use two spatial kernels. `l` is the periodic kernel lengthscale: fixed at 0.8 except for the 7-day / 1.2 setting.

The original 0.074 runs are sourced from `../../new_tests/may_5_11_2025` and `../../new_tests/nov_3_9_2025`; the remaining runs are from this `new_tests` folder. Saved parameters, rather than folder names, identify the settings. The unnamed September 13 runs use 7 days.

Using `days_per_standardized_unit` from the saved time diagnostics, 0.074 corresponds to:

| Window | Days per standardized unit | 0.074 in days |
| --- | ---: | ---: |
| May | 141.450521 | 10.467339 |
| November | 193.989476 | 14.355221 |

Values are copied from saved metrics (six decimals here; full precision in the CSV). **Bold** marks the best available value within each table; lower is better for MAE, RMSE and Wasserstein, higher for other metrics. `—` means a missing run, unsaved metric or null value; consult the CSV status and source columns. No missing metrics were recomputed.

First-three-day values are pooled metrics from the first three test dates of the same run, not a separate retraining experiment. Original 0.074 runs have no saved first-three-day metrics. Redistribution has only 0.074 and 7-day runs for these settings. The two May hard-classifier 14-day runs have identical full-week metrics; the latest run is used. The separate 10.47-day runs are excluded because they are not exactly 0.074.

## Full week

### May — LGCP 2 spatial kernels

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | -0.589872 | -0.645620 | -0.618982 | -0.604736 | -0.633330 | **-0.589330** | -0.651114 |
| mae_obs | **0.365010** | 0.398090 | 0.383453 | 0.372235 | 0.388098 | 0.370837 | 0.396899 |
| rmse_obs | **0.944014** | 1.018722 | 0.981945 | 0.983134 | 1.009807 | 0.973223 | 1.016652 |
| wasserstein | — | 0.276892 | 0.268852 | 0.258628 | 0.270796 | **0.255184** | 0.288200 |
| mean_ll_daily | **-8.407228** | -9.840040 | -8.745444 | -8.924344 | -9.352302 | -8.418312 | -9.682485 |
| mae_daily | 11.178913 | 12.786182 | 11.411906 | 11.597948 | 12.218785 | **10.817981** | 12.767623 |
| rmse_daily | **12.484439** | 14.633890 | 13.037310 | 13.438716 | 14.034124 | 12.737865 | 14.539611 |
| daily_delta_corr | 0.754329 | 0.283257 | 0.607336 | 0.690746 | 0.364861 | **0.795741** | 0.344203 |
| daily_direction_accuracy_moving | 0.500000 | **0.666667** | 0.500000 | 0.333333 | **0.666667** | 0.333333 | **0.666667** |
| accuracy | — | **0.160350** | **0.160350** | **0.160350** | **0.160350** | **0.160350** | **0.160350** |
| precision | — | **0.160350** | **0.160350** | **0.160350** | **0.160350** | **0.160350** | **0.160350** |
| recall | — | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |

### May — LGCP + hard classifier

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | **-1.722158** | -1.781741 | -1.753850 | -1.750914 | -1.773409 | -1.732727 | -1.790164 |
| mae_obs | **0.304812** | 0.332922 | 0.320497 | 0.310016 | 0.326354 | 0.307075 | 0.332417 |
| rmse_obs | **0.947190** | 1.020528 | 0.985422 | 0.986749 | 1.012165 | 0.977154 | 1.020018 |
| wasserstein | 0.214798 | 0.217116 | 0.208628 | 0.197912 | 0.211014 | **0.187570** | 0.224083 |
| mean_ll_daily | -42.531423 | -44.674444 | -42.900387 | -42.750481 | -43.851035 | **-42.108450** | -44.879849 |
| mae_daily | 12.292859 | 13.819784 | 12.489990 | 12.517263 | 13.227168 | **11.920936** | 13.922883 |
| rmse_daily | 14.098388 | 15.683882 | 14.408217 | 14.342570 | 15.086506 | **13.716863** | 15.861289 |
| daily_delta_corr | 0.848604 | 0.478132 | 0.749748 | 0.862458 | 0.566924 | **0.925349** | 0.534584 |
| daily_direction_accuracy_moving | 0.666667 | 0.500000 | 0.500000 | 0.666667 | 0.500000 | **0.833333** | 0.500000 |
| accuracy | **0.892128** | **0.892128** | **0.892128** | **0.892128** | **0.892128** | **0.892128** | **0.892128** |
| precision | **0.636364** | **0.636364** | **0.636364** | **0.636364** | **0.636364** | **0.636364** | **0.636364** |
| recall | **0.763636** | **0.763636** | **0.763636** | **0.763636** | **0.763636** | **0.763636** | **0.763636** |

### May — LGCP + redistribution

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | **-1.135519** | -1.177032 | — | — | — | — | — |
| mae_obs | **0.317176** | 0.352019 | — | — | — | — | — |
| rmse_obs | **0.968488** | 1.066296 | — | — | — | — | — |
| wasserstein | 0.167963 | **0.166565** | — | — | — | — | — |
| mean_ll_daily | **-8.407228** | -9.840041 | — | — | — | — | — |
| mae_daily | **11.178913** | 12.786182 | — | — | — | — | — |
| rmse_daily | **12.484439** | 14.633890 | — | — | — | — | — |
| daily_delta_corr | **0.754329** | 0.283257 | — | — | — | — | — |
| daily_direction_accuracy_moving | 0.500000 | **0.666667** | — | — | — | — | — |
| accuracy | **0.641399** | **0.641399** | — | — | — | — | — |
| precision | **0.292683** | **0.292683** | — | — | — | — | — |
| recall | **0.872727** | **0.872727** | — | — | — | — | — |

### November — LGCP 2 spatial kernels

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | -0.585129 | -0.547564 | -0.581027 | -0.572442 | -0.565757 | -0.569914 | **-0.546956** |
| mae_obs | 0.342784 | 0.322804 | 0.329333 | 0.325588 | **0.320900** | 0.334199 | 0.327065 |
| rmse_obs | 0.701959 | **0.681476** | 0.693031 | 0.701049 | 0.690513 | 0.698709 | 0.689957 |
| wasserstein | — | 0.249779 | 0.247163 | 0.242693 | 0.242795 | **0.242203** | 0.248261 |
| mean_ll_daily | **-2.866134** | -3.118149 | -3.036072 | -3.188052 | -3.182965 | -2.970106 | -3.038937 |
| mae_daily | **4.031605** | 4.628381 | 4.363532 | 4.909810 | 4.859246 | 4.414204 | 4.466452 |
| rmse_daily | **4.582466** | 5.603746 | 5.133198 | 5.639205 | 5.715332 | 5.029956 | 5.322801 |
| daily_delta_corr | 0.951185 | 0.921125 | 0.947161 | 0.939519 | **0.981130** | 0.938431 | 0.968026 |
| daily_direction_accuracy_moving | 0.800000 | **1.000000** | 0.800000 | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| accuracy | — | **0.154519** | **0.154519** | **0.154519** | **0.154519** | **0.154519** | **0.154519** |
| precision | — | **0.154519** | **0.154519** | **0.154519** | **0.154519** | **0.154519** | **0.154519** |
| recall | — | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |

### November — LGCP + hard classifier

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | -1.794147 | **-1.787007** | -1.800178 | -1.802910 | -1.796483 | -1.796775 | -1.789895 |
| mae_obs | 0.246165 | 0.241674 | 0.244931 | 0.249156 | 0.244360 | 0.249518 | **0.241205** |
| rmse_obs | 0.680988 | **0.669351** | 0.680686 | 0.690402 | 0.680079 | 0.683116 | 0.675673 |
| wasserstein | 0.185551 | 0.184882 | **0.180725** | 0.184604 | 0.184795 | 0.181847 | 0.185117 |
| mean_ll_daily | -15.376984 | -15.236836 | **-15.111195** | -15.345405 | -15.334948 | -15.254975 | -15.249464 |
| mae_daily | 9.080839 | 9.045640 | **8.845562** | 9.034538 | 9.046124 | 8.897238 | 9.055531 |
| rmse_daily | 10.525329 | 10.426081 | 10.306365 | 10.425602 | 10.402550 | **10.287948** | 10.449176 |
| daily_delta_corr | 0.867748 | 0.910632 | 0.884650 | 0.828318 | 0.851680 | 0.823563 | **0.927738** |
| daily_direction_accuracy_moving | **0.800000** | **0.800000** | 0.600000 | **0.800000** | **0.800000** | **0.800000** | **0.800000** |
| accuracy | **0.892128** | **0.892128** | **0.892128** | **0.892128** | **0.892128** | **0.892128** | **0.892128** |
| precision | **0.648148** | **0.648148** | **0.648148** | **0.648148** | **0.648148** | **0.648148** | **0.648148** |
| recall | **0.660377** | **0.660377** | **0.660377** | **0.660377** | **0.660377** | **0.660377** | **0.660377** |

### November — LGCP + redistribution

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | -1.607768 | **-1.603935** | — | — | — | — | — |
| mae_obs | 0.238396 | **0.232602** | — | — | — | — | — |
| rmse_obs | 0.591599 | **0.581972** | — | — | — | — | — |
| wasserstein | **0.106536** | 0.115240 | — | — | — | — | — |
| mean_ll_daily | **-2.866134** | -3.118149 | — | — | — | — | — |
| mae_daily | **4.031605** | 4.628381 | — | — | — | — | — |
| rmse_daily | **4.582466** | 5.603746 | — | — | — | — | — |
| daily_delta_corr | **0.951185** | 0.921125 | — | — | — | — | — |
| daily_direction_accuracy_moving | 0.800000 | **1.000000** | — | — | — | — | — |
| accuracy | **0.618076** | **0.618076** | — | — | — | — | — |
| precision | **0.243421** | **0.243421** | — | — | — | — | — |
| recall | **0.698113** | **0.698113** | — | — | — | — | — |

## First 3 days

### May — LGCP 2 spatial kernels

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | — | -0.784290 | -0.740544 | -0.718692 | -0.762936 | **-0.682363** | -0.794167 |
| mae_obs | — | 0.497789 | 0.460589 | 0.463570 | 0.482410 | **0.457047** | 0.497291 |
| rmse_obs | — | 1.181250 | 1.139655 | 1.146567 | 1.166422 | **1.124612** | 1.178320 |
| wasserstein | — | 0.304431 | 0.303629 | 0.287061 | 0.308404 | **0.275290** | 0.321811 |
| mean_ll_daily | — | -13.870221 | -11.823744 | -11.573464 | -13.128425 | **-10.720547** | -13.650021 |
| mae_daily | — | 17.238631 | 14.964193 | 14.313951 | 16.372457 | **13.008212** | 16.971765 |
| rmse_daily | — | 18.687214 | 16.644457 | 16.597781 | 17.946008 | **15.667402** | 18.366166 |
| daily_delta_corr | — | — | — | — | — | — | — |
| daily_direction_accuracy_moving | — | **0.500000** | **0.500000** | 0.000000 | **0.500000** | 0.000000 | **0.500000** |
| accuracy | — | **0.176871** | **0.176871** | **0.176871** | **0.176871** | **0.176871** | **0.176871** |
| precision | — | **0.176871** | **0.176871** | **0.176871** | **0.176871** | **0.176871** | **0.176871** |
| recall | — | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |

### May — LGCP + hard classifier

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | — | -1.180675 | -1.154494 | -1.121076 | -1.165799 | **-1.086638** | -1.192324 |
| mae_obs | — | 0.395213 | 0.367465 | 0.362908 | 0.384286 | **0.356045** | 0.396954 |
| rmse_obs | — | 1.162661 | 1.125803 | 1.128134 | 1.148457 | **1.107530** | 1.162112 |
| wasserstein | — | 0.225496 | 0.225494 | 0.195677 | 0.228249 | **0.179409** | 0.237693 |
| mean_ll_daily | — | -14.612619 | -12.328940 | -10.806216 | -13.552909 | **-9.715339** | -14.776427 |
| mae_daily | — | 16.963726 | 15.165691 | 14.231535 | 16.190914 | **13.148323** | 16.882212 |
| rmse_daily | — | 18.183807 | 16.812357 | 15.681768 | 17.565032 | **14.708279** | 18.245052 |
| daily_delta_corr | — | — | — | — | — | — | — |
| daily_direction_accuracy_moving | — | 0.500000 | 0.500000 | 0.500000 | 0.500000 | **1.000000** | 0.500000 |
| accuracy | — | **0.857143** | **0.857143** | **0.857143** | **0.857143** | **0.857143** | **0.857143** |
| precision | — | **0.560976** | **0.560976** | **0.560976** | **0.560976** | **0.560976** | **0.560976** |
| recall | — | **0.884615** | **0.884615** | **0.884615** | **0.884615** | **0.884615** | **0.884615** |

### May — LGCP + redistribution

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | — | **-1.175260** | — | — | — | — | — |
| mae_obs | — | **0.434452** | — | — | — | — | — |
| rmse_obs | — | **1.273912** | — | — | — | — | — |
| wasserstein | — | **0.150032** | — | — | — | — | — |
| mean_ll_daily | — | **-13.870221** | — | — | — | — | — |
| mae_daily | — | **17.238631** | — | — | — | — | — |
| rmse_daily | — | **18.687214** | — | — | — | — | — |
| daily_delta_corr | — | — | — | — | — | — | — |
| daily_direction_accuracy_moving | — | **0.500000** | — | — | — | — | — |
| accuracy | — | **0.857143** | — | — | — | — | — |
| precision | — | **0.560976** | — | — | — | — | — |
| recall | — | **0.884615** | — | — | — | — | — |

### November — LGCP 2 spatial kernels

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | — | -0.802258 | -0.863710 | -0.834922 | -0.825174 | -0.840560 | **-0.801310** |
| mae_obs | — | 0.458945 | 0.465120 | 0.465842 | **0.452182** | 0.481824 | 0.468737 |
| rmse_obs | — | 0.831696 | 0.849427 | 0.844994 | **0.825676** | 0.848792 | 0.842175 |
| wasserstein | — | 0.338183 | 0.339689 | 0.324640 | **0.318870** | 0.323180 | 0.337114 |
| mean_ll_daily | — | -3.767024 | -3.784716 | -3.516949 | -3.649010 | **-3.370695** | -3.644705 |
| mae_daily | — | 6.291252 | 6.750526 | 6.097293 | 6.226822 | **5.660812** | 6.080715 |
| rmse_daily | — | 7.119232 | 6.942546 | 6.414792 | 6.804966 | **5.967027** | 6.775541 |
| daily_delta_corr | — | — | — | — | — | — | — |
| daily_direction_accuracy_moving | — | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| accuracy | — | **0.231293** | **0.231293** | **0.231293** | **0.231293** | **0.231293** | **0.231293** |
| precision | — | **0.231293** | **0.231293** | **0.231293** | **0.231293** | **0.231293** | **0.231293** |
| recall | — | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |

### November — LGCP + hard classifier

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | — | **-2.804421** | -2.842652 | -2.814500 | -2.804755 | -2.814328 | -2.808058 |
| mae_obs | — | 0.364375 | 0.376184 | 0.371796 | **0.362270** | 0.377848 | 0.364861 |
| rmse_obs | — | 0.815007 | 0.841470 | 0.828784 | **0.810796** | 0.826120 | 0.821620 |
| wasserstein | — | 0.260425 | 0.266138 | 0.255110 | **0.252036** | 0.254856 | 0.266754 |
| mean_ll_daily | — | -8.440087 | -9.067090 | -8.295540 | **-8.028427** | -8.591725 | -8.841716 |
| mae_daily | — | 12.640387 | 12.935590 | 12.397917 | **12.275476** | 12.374222 | 12.918752 |
| rmse_daily | — | 12.848977 | 13.082285 | 12.581850 | **12.449850** | 12.595876 | 13.043536 |
| daily_delta_corr | — | — | — | — | — | — | — |
| daily_direction_accuracy_moving | — | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| accuracy | — | **0.816327** | **0.816327** | **0.816327** | **0.816327** | **0.816327** | **0.816327** |
| precision | — | **0.594595** | **0.594595** | **0.594595** | **0.594595** | **0.594595** | **0.594595** |
| recall | — | **0.647059** | **0.647059** | **0.647059** | **0.647059** | **0.647059** | **0.647059** |

### November — LGCP + redistribution

| Metric | 0.074 | 7 days | 10 days | 14 days | 7 days, l=1.2 | 14 days init, trainable | 7 days init, trainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_ll_obs | — | **-2.768397** | — | — | — | — | — |
| mae_obs | — | **0.354770** | — | — | — | — | — |
| rmse_obs | — | **0.742477** | — | — | — | — | — |
| wasserstein | — | **0.156340** | — | — | — | — | — |
| mean_ll_daily | — | **-3.767023** | — | — | — | — | — |
| mae_daily | — | **6.291253** | — | — | — | — | — |
| rmse_daily | — | **7.119232** | — | — | — | — | — |
| daily_delta_corr | — | — | — | — | — | — | — |
| daily_direction_accuracy_moving | — | **1.000000** | — | — | — | — | — |
| accuracy | — | **0.816327** | — | — | — | — | — |
| precision | — | **0.594595** | — | — | — | — | — |
| recall | — | **0.647059** | — | — | — | — | — |

## Learned periods

The trainable settings start at 7 or 14 calendar days with periodic lengthscale fixed at 0.8. The standardized period is optimized during training; the requested days specify its initialization, not its final value. Final days below equal the saved final period multiplied by `days_per_standardized_unit`.

| Window | Model | Initial period (days) | Final standardized period | Final period (days) |
| --- | --- | ---: | ---: | ---: |
| May | LGCP + hard classifier | 14 | 0.099192 | 14.030812 |
| May | LGCP + hard classifier | 7 | 0.049495 | 7.001152 |
| May | LGCP 2 spatial kernels | 14 | 0.099192 | 14.030812 |
| May | LGCP 2 spatial kernels | 7 | 0.049495 | 7.001152 |
| November | LGCP + hard classifier | 14 | 0.072250 | 14.015741 |
| November | LGCP + hard classifier | 7 | 0.036111 | 7.005183 |
| November | LGCP 2 spatial kernels | 14 | 0.072250 | 14.015741 |
| November | LGCP 2 spatial kernels | 7 | 0.036111 | 7.005183 |

## Source runs

| Window | Model | Setting | Run |
| --- | --- | --- | --- |
| May | LGCP + hard classifier | 0.074 | [2024+2025_may_5_11_lgcp_spat_rbf_l0.2_rbf_l0.01_classifier_20260910-160709](../../new_tests/may_5_11_2025/2024+2025_may_5_11_lgcp_spat_rbf_l0.2_rbf_l0.01_classifier_20260910-160709/params.yaml) |
| May | LGCP + hard classifier | 10 days | [2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_10_days_l0.8_20260914-112920](2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_10_days_l0.8_20260914-112920/params.yaml) |
| May | LGCP + hard classifier | 14 days | [2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_14_days_20260914-095841](2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_14_days_20260914-095841/params.yaml) |
| May | LGCP + hard classifier | 14 days init, trainable | [2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_init_14_days_l0.8_trainable_20260914-153544](2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_init_14_days_l0.8_trainable_20260914-153544/params.yaml) |
| May | LGCP + hard classifier | 7 days | [2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_20260913-170110](2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_20260913-170110/params.yaml) |
| May | LGCP + hard classifier | 7 days init, trainable | [2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_init_7_days_l0.8_trainable_20260914-155702](2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_init_7_days_l0.8_trainable_20260914-155702/params.yaml) |
| May | LGCP + hard classifier | 7 days, l=1.2 | [2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_7_days_l1.2_20260914-105553](2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_7_days_l1.2_20260914-105553/params.yaml) |
| May | LGCP + redistribution | 0.074 | [2024+2025_may_5_11_lgcp_spat_rbf_l0.2_rbf_l0.01_classifier_redistribute_20260910-162520](../../new_tests/may_5_11_2025/2024+2025_may_5_11_lgcp_spat_rbf_l0.2_rbf_l0.01_classifier_redistribute_20260910-162520/params.yaml) |
| May | LGCP + redistribution | 7 days | [2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_redistribuite_20260913-171131](2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_redistribuite_20260913-171131/params.yaml) |
| May | LGCP 2 spatial kernels | 0.074 | [2024+2025_may_5_11_lgcp_spat_rbf_l0.2_rbf_l0.01_periodic_l0.8_20260908-122430](../../new_tests/may_5_11_2025/2024+2025_may_5_11_lgcp_spat_rbf_l0.2_rbf_l0.01_periodic_l0.8_20260908-122430/params.yaml) |
| May | LGCP 2 spatial kernels | 10 days | [2024+2025_may_5_11_lgcp_2_spatial_kernels_period_10_days_l0.8_20260914-112007](2024+2025_may_5_11_lgcp_2_spatial_kernels_period_10_days_l0.8_20260914-112007/params.yaml) |
| May | LGCP 2 spatial kernels | 14 days | [2024+2025_may_5_11_lgcp_2_spatial_kernels_period_14_days_20260914-095010](2024+2025_may_5_11_lgcp_2_spatial_kernels_period_14_days_20260914-095010/params.yaml) |
| May | LGCP 2 spatial kernels | 14 days init, trainable | [2024+2025_may_5_11_lgcp_2_spatial_kernels_period_init_14_days_l0.8_trainable_20260914-152821](2024+2025_may_5_11_lgcp_2_spatial_kernels_period_init_14_days_l0.8_trainable_20260914-152821/params.yaml) |
| May | LGCP 2 spatial kernels | 7 days | [2024+2025_may_5_11_lgcp_2_spatial_kernels_20260913-165051](2024+2025_may_5_11_lgcp_2_spatial_kernels_20260913-165051/params.yaml) |
| May | LGCP 2 spatial kernels | 7 days init, trainable | [2024+2025_may_5_11_lgcp_2_spatial_kernels_period_init_7_days_l0.8_trainable_20260914-154931](2024+2025_may_5_11_lgcp_2_spatial_kernels_period_init_7_days_l0.8_trainable_20260914-154931/params.yaml) |
| May | LGCP 2 spatial kernels | 7 days, l=1.2 | [2024+2025_may_5_11_lgcp_2_spatial_kernels_period_7_days_l1.2_20260914-104655](2024+2025_may_5_11_lgcp_2_spatial_kernels_period_7_days_l1.2_20260914-104655/params.yaml) |
| November | LGCP + hard classifier | 0.074 | [2024+2025_nov_3_9_lgcp_spat_rbf_l0.2_rbf_l0.01_classifier_20260910-160302](../../new_tests/nov_3_9_2025/2024+2025_nov_3_9_lgcp_spat_rbf_l0.2_rbf_l0.01_classifier_20260910-160302/params.yaml) |
| November | LGCP + hard classifier | 10 days | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_10_days_l0.8_20260914-112441](2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_10_days_l0.8_20260914-112441/params.yaml) |
| November | LGCP + hard classifier | 14 days | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_14_days_20260914-095421](2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_14_days_20260914-095421/params.yaml) |
| November | LGCP + hard classifier | 14 days init, trainable | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_init_14_days_l0.8_trainable_20260914-153202](2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_init_14_days_l0.8_trainable_20260914-153202/params.yaml) |
| November | LGCP + hard classifier | 7 days | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_20260913-165556](2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_20260913-165556/params.yaml) |
| November | LGCP + hard classifier | 7 days init, trainable | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_init_7_days_l0.8_trainable_20260914-155314](2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_init_7_days_l0.8_trainable_20260914-155314/params.yaml) |
| November | LGCP + hard classifier | 7 days, l=1.2 | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_7_days_l1.2_20260914-105128](2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_7_days_l1.2_20260914-105128/params.yaml) |
| November | LGCP + redistribution | 0.074 | [2024+2025_nov_3_9_lgcp_spat_rbf_l0.2_rbf_l0.01_classifier_redistribute_20260910-162054](../../new_tests/nov_3_9_2025/2024+2025_nov_3_9_lgcp_spat_rbf_l0.2_rbf_l0.01_classifier_redistribute_20260910-162054/params.yaml) |
| November | LGCP + redistribution | 7 days | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_redistribuite_20260913-170629](2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_redistribuite_20260913-170629/params.yaml) |
| November | LGCP 2 spatial kernels | 0.074 | [2024+2025_nov_3_9_lgcp_spat_rbf_l0.2_rbf_l0.01_periodic_l0.8_20260908-124922](../../new_tests/nov_3_9_2025/2024+2025_nov_3_9_lgcp_spat_rbf_l0.2_rbf_l0.01_periodic_l0.8_20260908-124922/params.yaml) |
| November | LGCP 2 spatial kernels | 10 days | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_10_days_l0.8_20260914-111558](2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_10_days_l0.8_20260914-111558/params.yaml) |
| November | LGCP 2 spatial kernels | 14 days | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_14_days_20260914-094624](2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_14_days_20260914-094624/params.yaml) |
| November | LGCP 2 spatial kernels | 14 days init, trainable | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_init_14_days_l0.8_trainable_20260914-152447](2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_init_14_days_l0.8_trainable_20260914-152447/params.yaml) |
| November | LGCP 2 spatial kernels | 7 days | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_20260913-164607](2024+2025_nov_3_9_lgcp_2_spatial_kernels_20260913-164607/params.yaml) |
| November | LGCP 2 spatial kernels | 7 days init, trainable | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_init_7_days_l0.8_trainable_20260914-154553](2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_init_7_days_l0.8_trainable_20260914-154553/params.yaml) |
| November | LGCP 2 spatial kernels | 7 days, l=1.2 | [2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_7_days_l1.2_20260914-104258](2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_7_days_l1.2_20260914-104258/params.yaml) |
