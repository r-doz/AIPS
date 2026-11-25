from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import pandas as pd
import numpy as np
from src.models.metrics_lgcp import evaluate_metrics
root=Path('reports/seeded_monthly_windows_salinity_oldclassifier/11_train_lgcp/seed_1')
frames=[];daily=[]
for p in sorted((root/'runs').glob('2025_november_*/*/spatial_predictions.csv')):
 df=pd.read_csv(p);df['source_window']=p.parent.parent.name;frames.append(df)
 for date,g in df.groupby('date'):
  m=evaluate_metrics(g.observed.to_numpy(),g.predicted.to_numpy(),g.date)
  daily.append(dict(date=date,source_window=p.parent.parent.name,observed_total=g.observed.sum(),predicted_total=g.predicted.sum(),active_cells=int((g.observed>0).sum()),**m))
d=pd.DataFrame(daily).sort_values('date').set_index('date')
w=d[d.source_window=='2025_november_15_21'];target=d.loc['2025-11-20']
keys=['mae_obs','rmse_obs','mae_daily','wasserstein','f1','accuracy','observed_total','active_cells','mean_ll_obs']
print('DAILY COMPARISONS (ascending ranks, average ties)')
for key in keys:
 print(key,'target',target[key])
 for label,g in [('November',d),('week',w)]:
  print(label,'median',g[key].median(),'mean',g[key].mean(),'rank',g[key].rank().loc['2025-11-20'],'of',len(g),'nearest median',g[key].sub(g[key].median()).abs().nsmallest(3).to_dict())
print('\nWEEK DAILY\n',w[['observed_total','predicted_total','mae_obs','rmse_obs','mae_daily','wasserstein','f1']].to_string())
weekly=pd.read_csv(root/'per_window_metrics.csv');weekly=weekly[weekly.month=='november'].set_index('name')
print('\nWEEK METRICS')
for key in keys:
 if key not in weekly:continue
 s=weekly[key];print(key,'target',s.loc['2025_november_15_21'],'median',s.median(),'rank',s.rank().loc['2025_november_15_21'],'closest',s.sub(s.median()).abs().idxmin())
# Verify recomputed pooled metrics reproduce the experiment reports.
for name,g in pd.concat(frames).groupby('source_window'):
 metrics=evaluate_metrics(g.observed.to_numpy(),g.predicted.to_numpy(),g.date)
 for key,value in metrics.items():
  if key in weekly:
   np.testing.assert_allclose(value,weekly.loc[name,key],rtol=1e-10,atol=1e-10)
print('PASS pooled metrics reproduce all four saved weekly reports')
out=Path('reports/paper/map_design_61/november_20_representativeness');out.mkdir(parents=True,exist_ok=True)
d.drop(columns=['daily_delta_corr','daily_direction_accuracy_moving'],errors='ignore').to_csv(out/'daily_metrics.csv')
weekly.to_csv(out/'weekly_metrics.csv')
