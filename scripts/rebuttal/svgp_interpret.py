"""Run reports/interpretability_hurdle/interpret_hurdle.py (Table 6, App. B.7) on another set of 80 runs.

The analysis code is executed unchanged; only its input folder (RUNS) and output folder (OUT) are
replaced. The posterior-mean field A m it decomposes is the same under the SVGP marginal, whose
residual has zero mean.

Usage: python svgp_interpret.py <runs dir with seed_*/runs/*/*/model.pt> <output dir>
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports/interpretability_hurdle/interpret_hurdle.py"
# checkouts without that file (Demetra) use a verbatim copy; it runs with SRC as __file__, so its ROOT is the repo
COPY = ROOT / "reports/rebuttal/svgp_residual/interpret_hurdle.py"
OLD_RUNS = 'RUNS = ROOT / "reports/seeded_monthly_windows_hurdle"'
OLD_OUT = 'OUT = ROOT / "reports/interpretability_hurdle"'

runs, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
code = (SRC if SRC.exists() else COPY).read_text()
assert OLD_RUNS in code and OLD_OUT in code, "interpret_hurdle.py changed; update the replacements"
code = code.replace(OLD_RUNS, f"RUNS = Path({str(runs)!r})").replace(OLD_OUT, f"OUT = Path({str(out)!r})")
exec(compile(code, str(SRC), "exec"), {"__name__": "__main__", "__file__": str(SRC)})
