# Red-LGCP: forecasting daily fishing activity with a hurdle log-Gaussian Cox process

This repository contains the code, configurations and final results for forecasting daily fishing
activity on a grid of 49 coastal cells in the northern Adriatic Sea. The target is the daily count of
AIS apparent-fishing records per cell; the covariates are chlorophyll-a, sea temperature and salinity
(Copernicus Marine), calendar and regulatory indicators (weekends, public holidays, fishing ban) and
the counts observed 1 and 7 days earlier.

**Red-LGCP** couples a sparse variational log-Gaussian Cox process (LGCP) with a neural activity
classifier. The two are trained together through a hurdle likelihood: the classifier models whether
a cell is active, and the LGCP how much activity an active cell hosts. Point predictions use a gate
whose threshold accounts for the classifier's Monte Carlo dropout uncertainty, followed by a
redistribution of the mass removed from rejected cells.

All models are evaluated on 8 weekly test windows (days 1-7, 8-14, 15-21 and 22-28 of May and
November 2025), retraining on all data before each window, with 10 seeds for the stochastic models.

## Repository structure

```
config/        YAML configurations: data pipeline and final experiments
scripts/       data pipeline (01-04), training and evaluation (11-18), experiment runners,
               hyperparameter re-validation
src/data/      download, cleaning and integration of the data sources
src/models/    LGCP, hurdle LGCP, baselines, activity gate and metrics
src/visualization/  plotting utilities
reports/       final results and the scripts that produce the paper's tables and figures
notebooks/     map figure of the final predictions
docs/          detailed documentation of the data pipeline
tests/         unit tests
data/          raw, intermediate and processed data (processed files are not tracked)
```

## Installation

The code was developed with Python 3.13. Install the pinned dependencies with

```bash
pip install -r requirements.txt
```

All experiments run on CPU (`device: cpu` in the configurations).

## Data

### Credentials

Downloading the raw data requires three accounts. Credentials are read from the git-ignored
`secrets/` folder or from the Copernicus Marine client configuration.

- **Copernicus Marine Service**: create an account at https://marine.copernicus.eu and run
  `copernicusmarine login` once.
- **Global Fishing Watch API** (AIS fishing effort): create a token at
  https://globalfishingwatch.org/our-apis/ and save it as `secrets/gfw_token.txt`.
- **Google BigQuery** (Global Fishing Watch Sentinel-2 vessel detections): in a Google Cloud
  project, enable the BigQuery API, create a service account with BigQuery access and save its JSON
  key as `secrets/gcp_bigquery_key.json`.

### Building the dataset

The study area, dates and source products are set in `config/data.yaml`. The scripts process one
year at a time, so run steps 1-3 with `year: 2024` and again with `year: 2025`, then add salinity
for both years:

```bash
python scripts/01_get_data.py        # download Copernicus, GFW AIS and Sentinel-2 data
python scripts/02_clean_data.py      # clean and restrict to the study area
python scripts/03_build_dataset.py   # merge into data/processed/cpr_gfw_{year}.parquet
python scripts/04_add_salinity.py    # add salinity -> data/processed/cpr_gfw_salinity_{year}.parquet
```

The resulting files contain one row per cell and day (49 cells x 731 days) with the counts, the
environmental covariates and the calendar indicators; the lagged counts are computed when the data
are loaded for training. See [docs/data_pipeline.md](docs/data_pipeline.md) for the full description of
the sources, the cleaning steps and the known provenance issues.

## Running the experiments

Every training script takes a configuration file and evaluates one test window. The runners repeat it
over the 8 weekly windows (`scripts/run_monthly_windows.py`) and over several seeds
(`scripts/run_seeded_monthly_windows.py`), and aggregate the per-window metrics. For example, the
final model:

```bash
python scripts/run_seeded_monthly_windows.py \
    --script scripts/11c_train_hurdle_lgcp.py \
    --config config/exp4_salinity_hurdle.yaml \
    --seeds 1-10 --parallel 3 \
    --output-dir reports/seeded_monthly_windows_hurdle
```

| Model | Script | Configuration | Results |
|---|---|---|---|
| Red-LGCP (final) | `11c_train_hurdle_lgcp.py` | `exp4_salinity_hurdle.yaml` | `reports/seeded_monthly_windows_hurdle` |
| Last Available Poisson (LAP) | `13_evaluate_last_available_poisson.py` | `13_evaluate_last_available_poisson.yaml` | `reports/seeded_monthly_windows/13_evaluate_last_available_poisson` |
| Global Cell Mean Poisson (GCMP) | `12b_global_cell_mean.py` | `12b_global_cell_mean.yaml` | `reports/seeded_monthly_windows/12b_global_cell_mean` |
| Window Poisson GLM (WGLM) | `14_train_window_poisson_glm.py` | `14_window_poisson_glm_salinity.yaml` | `reports/seeded_monthly_windows_salinity/14_train_window_poisson_glm` |
| GNN | `15_train_gnn.py` | `15_gnn_salinity_cellpoisson.yaml` | `reports/seeded_monthly_windows_salinity_gnn_cellpoisson/15_train_gnn` |
| ConvLSTM | `18_train_convlstm.py` | `18_convlstm_salinity.yaml` | `reports/seeded_monthly_windows_salinity/18_train_convlstm` |
| Ablation: w/o kernel | `11c_train_hurdle_lgcp.py` | `exp4_salinity_hurdle_no_kernel.yaml` | `reports/seeded_monthly_windows_hurdle_ablation/no_kernel` |
| Ablation: two-stage | `11_train_lgcp.py` | `exp4_salinity.yaml` | `reports/seeded_monthly_windows_salinity/11_train_lgcp` |
| Ablation: w/o classifier | `11_train_lgcp.py` | `exp4_salinity_no_classifier.yaml` | `reports/seeded_monthly_windows_ablation/no_classifier` |

LAP, GCMP and WGLM are deterministic and were run with a single seed (`--seeds 1`). The ablations
without uncertainty, without redistribution and without gate need no retraining: they are computed
from the predictions saved by the Red-LGCP runs (`hurdle_predictions.csv` in each run folder).

Each run folder contains the configuration (`params.yaml`) and the test metrics (`metrics*.yaml`) and,
depending on the model, its predictions and checkpoint (`model.pt` or `model_state_dict.pt`). The
per-run plots were removed to keep the repository small; the training scripts regenerate them.

## Reproducing the tables and figures

The following scripts read the saved results and do not retrain any model. Run them from the
repository root, in this order; every table is written as a CSV file:

```bash
python reports/seeded_monthly_windows_hurdle_analysis.py      # pools the Red-LGCP seeds
python reports/probabilistic_scores/compute_scores.py         # log-likelihood and CRPS of all methods
python reports/statistical_tests/make_main_table.py           # main results table
python reports/statistical_tests/paired_window_tests.py       # paired Wilcoxon tests
python reports/probabilistic_scores/ablation_scores.py --stage base
python reports/probabilistic_scores/ablation_scores.py --stage nokernel
python reports/probabilistic_scores/ablation_scores.py --stage table   # ablation table
python reports/sensitivity_threshold/threshold_sensitivity.py # sensitivity to the gate threshold
python reports/interpretability_hurdle/interpret_hurdle.py    # LGCP parameters and latent field
python reports/interpretability_hurdle/make_table.py          # parameter table
```

| Table | File |
|---|---|
| Main results (mean and std per method and month) | `reports/statistical_tests/main_results_table.csv` |
| Paired tests, point metrics | `reports/statistical_tests/paired_window_tests_hurdle.csv` |
| Paired tests, log-likelihood and CRPS | `reports/probabilistic_scores/paired_tests.csv` |
| Ablation | `reports/statistical_tests/ablation_table.csv` |
| Paired tests, ablation | `reports/probabilistic_scores/ablation_paired_tests.csv` |
| Sensitivity to the gate threshold | `reports/sensitivity_threshold/summary.csv` |
| LGCP parameters | `reports/interpretability_hurdle/lgcp_parameters_table.csv` |

The maps of the predictions are produced by `notebooks/62_plot_map_design_last_model.ipynb`.

## Hyperparameter selection

Hyperparameters were re-validated on the two weeks preceding each May and November reporting week
(21 April - 4 May and 20 October - 2 November 2025) with:

```bash
python scripts/lgcp_lr_sweep_2weeks_before_salinity.py              # LGCP learning rate
python scripts/classifier_hparam_rerun_2weeks_before_salinity.py    # classifier of the two-stage model
python scripts/convlstm_hparam_rerun_2weeks_before_salinity.py      # ConvLSTM
python scripts/51c_baseline_hparam_rerun_2weeks_before_salinity.py  # Window Poisson GLM and GNN
```

The results and the selected values are in
`reports/paper/hparam_revalidation_2weeks_before_salinity/*/selected_hyperparameters.yaml`. The final
LGCP learning rate and the GLM, GNN and ConvLSTM settings correspond to these selections.

## Tests

```bash
python -m unittest discover -s tests
```

## Data sources

- Copernicus Marine Service, Mediterranean Sea physics and biogeochemistry products (sea temperature,
  salinity, chlorophyll-a).
- Global Fishing Watch, AIS-based apparent fishing effort.
- Global Fishing Watch, vessel detections from Sentinel-2 imagery (Zenodo, doi:10.5281/zenodo.15978308).

Please cite the data providers as indicated in the paper when using the data.
