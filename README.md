# Overview 
Analysis and forecasting of the fishing vessel in the Trieste Gulf. 

# How to use it 

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