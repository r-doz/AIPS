import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import pandas as pd
import torch
import yaml

sys.path.insert(0,str(Path.cwd()))
spec=importlib.util.spec_from_file_location('gnn_training','scripts/15_train_gnn.py')
gnn=importlib.util.module_from_spec(spec);spec.loader.exec_module(gnn)
parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True);parser.add_argument('--root',required=True);args=parser.parse_args()
torch.set_num_threads(1)
root=Path(args.root)
base=yaml.safe_load((root/'base_config.yaml').read_text())
windows=yaml.safe_load((root/'windows.yaml').read_text())
rows=[]
for w in windows:
 cfg=deepcopy(base)
 cfg.update(test_start_date=w['start'],test_end_date=w['end'],split_strategy='fixed_test_window',report_root=str(root),model_family=args.mode,run_name=w['name'],generate_spatial_plots=False,loss=dict(mode=args.mode,daily_weight=1.0))
 print(f'=== {args.mode}: {w["name"]} ===',flush=True)
 gnn.main(cfg)
 folder=root/args.mode/w['name']
 m=yaml.safe_load((folder/'metrics.yaml').read_text())
 rows.append(dict(mode=args.mode,window=w['name'],start=w['start'],end=w['end'],mae_daily=m['mae_daily'],rmse_daily=m['rmse_daily']))
 pd.DataFrame(rows).to_csv(root/f'{args.mode}_metrics.csv',index=False)
 print(f'WINDOW COMPLETE {w["name"]}: MAE={m["mae_daily"]:.4f}',flush=True)
print(f'ALL COMPLETE {args.mode}: mean MAE={pd.DataFrame(rows).mae_daily.mean():.4f}',flush=True)
