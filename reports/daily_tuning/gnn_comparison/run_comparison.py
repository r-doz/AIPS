import importlib.util
import sys
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import torch
sys.path.insert(0, str(Path.cwd()))
spec=importlib.util.spec_from_file_location('gnn_training', 'scripts/15_train_gnn.py')
gnn=importlib.util.module_from_spec(spec);spec.loader.exec_module(gnn)
torch.set_num_threads(1)
tuning=yaml.safe_load(Path('config/tune_daily_count_catboost_rmse.yaml').read_text())
cat_root=Path('reports/tuning/daily_count_catboost_rmse/20260909-185058-174431')
cat_params=yaml.safe_load((cat_root/'params.yaml').read_text())
windows=cat_params['tuning']['windows']
cat_best=yaml.safe_load((cat_root/'best_parameters.yaml').read_text())
cat_scores=pd.read_csv(cat_root/'window_metrics.csv')
cat_scores=cat_scores[cat_scores.trial==cat_best['trial']].set_index('window')
root=Path('reports/daily_tuning/gnn_comparison')/datetime.now().strftime('%Y%m%d-%H%M%S')
root.mkdir(parents=True)
base=yaml.safe_load(Path('config/15_gnn.yaml').read_text())
(root/'base_gnn_config.yaml').write_text(yaml.safe_dump(base,sort_keys=False))
(root/'catboost_config.yaml').write_text((cat_root/'best_config.yaml').read_text())
rows=[]
for w in windows:
    cfg=gnn.load_config('config/15_gnn.yaml')
    cfg.update(test_start_date=w['start'],test_end_date=w['end'],split_strategy='fixed_test_window',
               report_root=str(root),model_family='runs',run_name=w['name'],generate_spatial_plots=False)
    print(f'\n=== GNN window {w["name"]} ===',flush=True)
    gnn.main(cfg)
    folder=root/'runs'/w['name']
    metrics=yaml.safe_load((folder/'metrics.yaml').read_text())
    cat=cat_scores.loc[w['name']]
    pred=pd.read_csv(folder/'daily_predictions.csv')
    cp=pd.read_csv(cat_root/'validation_predictions.csv')
    cp=cp[(cp.trial==cat_best['trial']) & (cp.window==w['name'])]
    assert pd.to_datetime(pred.date).tolist()==pd.to_datetime(cp.date).tolist()
    np.testing.assert_allclose(np.abs(cp.observed_total.to_numpy()-pred.predicted_total.to_numpy()).mean(), metrics['mae_daily'],rtol=1e-5)
    rows.append(dict(window=w['name'],start=w['start'],end=w['end'],gnn_mae=metrics['mae_daily'],
                     catboost_mae=float(cat.mae_daily),gnn_rmse=metrics['rmse_daily'],catboost_rmse=float(cat.rmse_daily),
                     mae_difference_catboost_minus_gnn=float(cat.mae_daily)-metrics['mae_daily']))
    pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
frame=pd.DataFrame(rows)
summary=frame[['gnn_mae','catboost_mae','gnn_rmse','catboost_rmse']].mean().to_dict()
summary['catboost_mae_wins']=int((frame.catboost_mae<frame.gnn_mae).sum())
summary['n_windows']=len(frame)
(root/'summary.yaml').write_text(yaml.safe_dump(summary))
report=['# Current GNN versus tuned CatBoost RMSE','',
'Both use expanding training history and one-step-ahead observed lags. GNN keeps config/15_gnn.yaml architecture, inputs, 300 epochs and seed 0; CPU threads were set to 1. Spatial maps were omitted.',
'', 'CatBoost was selected on these eight windows, so this is a validation comparison, not an independent test.',
'', 'Input difference: GNN additionally uses cell_roll_mean_7, daily_total_lag_1, daily_total_lag_7, daily_total_roll_mean_7. This is not an identical-feature comparison.',
'', '| Window | GNN MAE | CatBoost MAE | GNN RMSE | CatBoost RMSE |','|---|---:|---:|---:|---:|']
for r in rows:
    report.append(f"| {r['window']} | {r['gnn_mae']:.3f} | {r['catboost_mae']:.3f} | {r['gnn_rmse']:.3f} | {r['catboost_rmse']:.3f} |")
report.append(f"| **Mean** | **{summary['gnn_mae']:.3f}** | **{summary['catboost_mae']:.3f}** | **{summary['gnn_rmse']:.3f}** | **{summary['catboost_rmse']:.3f}** |")
report+=['',f"CatBoost wins on daily MAE in {summary['catboost_mae_wins']}/8 windows.",f'CatBoost source: {cat_root}, trial {cat_best["trial"]}.']
(root/'report.md').write_text('\n'.join(report)+'\n')
print('\nCOMPARISON COMPLETE:',root,summary,flush=True)
