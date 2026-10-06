"""LaTeX table of the LGCP parameters of the joint hurdle Red-LGCP (from params_effects_all_runs.csv)."""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
d = pd.read_csv(OUT / "params_effects_all_runs.csv")
MONTHS = ("may", "november")


def r(x, nd):
    s = str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def value(col, nd=3):
    return [f"{r(d[d.month == m][col].mean(), nd)} $\\pm$ {r(d[d.month == m][col].std(ddof=0), nd)}" for m in MONTHS]


def effect(col):
    return [f"$\\times${r(d[d.month == m][col].mean(), 3)}" for m in MONTHS]


def share(col):
    return [f"{r(100 * d[d.month == m][col].mean(), 1)}\\%" for m in MONTHS]


rows = [("$\\alpha$", "Intercept", value("alpha"), ["", ""])]
for c, sym, desc in [("chl", "chl", "Chlorophyll-a (per SD)"), ("thetao", "temp", "Sea temperature (per $^\\circ$C)"),
                     ("salinity", "sal", "Salinity (per psu)"), ("fishing_block", "ban", "Fishing ban (on vs.\\ off)"),
                     ("is_holiday", "hol", "Public holiday (on vs.\\ off)"), ("is_weekend", "wkend", "Weekend (on vs.\\ off)"),
                     ("cell_lag_1", "lag1", "Count at $t-1$ (per count)"), ("cell_lag_7", "lag7", "Count at $t-7$ (per count)")]:
    rows.append((f"$\\beta_{{\\mathrm{{{sym}}}}}$", desc, value(f"beta_{c}"), effect(f"effect_{c}")))
kernel = [("$\\sigma^2_{s,\\mathrm{broad}}$", "Spatial RBF, $\\ell=0.2$", value("var_s_broad")),
          ("$\\sigma^2_{s,\\mathrm{loc}}$", "Spatial RBF, $\\ell=0.01$", value("var_s_loc")),
          ("$\\sigma^2_{t,\\mathrm{RBF}}$", "Temporal RBF, $\\ell=0.01$", value("var_t_rbf")),
          ("$\\sigma^2_{t,\\mathrm{per}}$", "Periodic, $\\ell=0.8$", value("var_t_per")),
          ("$p$", "Period (days)", value("period_days", 2))]
shares = [("Static", "Time-constant cell map", share("share_static")),
          ("Periodic", "Periodic component", share("share_cycle")),
          ("Short-term", "Temporal RBF component", share("share_short"))]

lines = [r"\multicolumn{6}{l}{\textit{Intensity of active cells} $\log\lambda=\alpha+\mathbf{x}^\top\boldsymbol{\beta}+f$} \\"]
lines += [f"{p} & {desc} & {v[0]} & {v[1]} & {e[0]} & {e[1]} \\\\" for p, desc, v, e in rows]
lines += [r"\midrule", r"\multicolumn{6}{l}{\textit{Kernel variances and period}} \\"]
lines += [f"{p} & {desc} & {v[0]} & {v[1]} & & \\\\" for p, desc, v in kernel]
lines += [r"\midrule", r"\multicolumn{6}{l}{\textit{Share of the variance of the posterior-mean latent field} $\bar f$} \\"]
lines += [f"{p} & {desc} & {v[0]} & {v[1]} & & \\\\" for p, desc, v in shares]

tex = r"""\begin{table}[t]
\centering
\caption{Parameters of the LGCP intensity of Red-LGCP (mean $\pm$ population standard deviation over 10 seeds $\times$ 4 weekly windows per month). The effect columns give the multiplicative effect on the intensity $\lambda$ of active cells, computed per run with the training statistics of its window. Since the kernel is a product of a spatial and a temporal kernel, the spatial and the temporal variances are identified only up to a common factor. The bottom block gives the share of the variance of $\bar f$ over all training cell-days carried by each component.}
\label{tab:lgcp_params}
\vspace{2pt}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}
\toprule
 & & \multicolumn{2}{c}{Value} & \multicolumn{2}{c}{Effect on $\lambda$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Parameter & Description & May & Nov. & May & Nov. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "lgcp_parameters_table.tex").write_text(tex)
print(tex)
