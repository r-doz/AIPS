# Overview 
Analysis and forecasting of the fishing vessel in the Trieste Gulf. 

# Keys and CSVs

### 0. Clone the folder 

Pay attention you get the "ts_gulf_coords.csv".

### 1. Generate keys 

Global Fishing Watch API key
- Site: https://globalfishingwatch.org/our-apis/ 
- Generate a TOKEN
- Call it "gfw_token.txt"
- Move it in the folder "secrets"

Google Cloud BigQuery service account key
- Site: https://console.cloud.google.com/welcome?project=double-catfish-475813-h4 (log in)
  - Select or create a project
  - Enable the BigQuery API
  - Create a service account
  - Grant permissione at the service account 
  - Create a JSON key
- Call it "gcp_bigquery_key.json"
- Move it in the folder secrets 

# Folder organization 

...

# Piepeline

### Data p
- set config/data
- run scripts/01_get_data
- run scripts/02_clean_data
- run scripts/03_build_dataset

### Models
Multi-kernel LGCP
- set config/train_basic_lgcp_multi_kernel
- run scripts/11_train_lgcp.py   --config config/train_basic_lgcp_multi_kernel.yaml

Global mean Poisson 
- set config/train_basic_lgcp_multi_kernel
- run scripts/12_evaluate_global_mean_poisson.py   --config config/train_basic_lgcp_multi_kernel.yaml

Global cell mean Poisson 
- set config/train_basic_lgcp_multi_kernel
- run scripts/12b_global_cell_mean.py   --config config/train_basic_lgcp_multi_kernel.yaml

Last available Poisson (one_step_ahead and frozen_train)
- set config/train_basic_lgcp_multi_kernel
- run scripts/13_evaluate_last_available_poisson.py    --config config/train_basic_lgcp_multi_kernel.yaml

Window Poisson GLM
- set config/14_window_poisson_glm.yaml
- run scripts/14_train_window_poisson_glm.py