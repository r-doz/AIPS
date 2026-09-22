import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.run_monthly_windows import aggregate_metrics, build_windows


class MonthlyWindowRunnerTests(unittest.TestCase):
    def test_builds_four_weeks_for_each_month_without_mutating_config(self):
        config = {"test_start_date": "2025-01-01", "run_name": "base"}
        jobs = build_windows(config, 2025)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(jobs[0]["test_start_date"], "2025-05-01")
        self.assertEqual(jobs[3]["test_end_date"], "2025-05-28")
        self.assertEqual(jobs[4]["test_start_date"], "2025-11-01")
        self.assertEqual(jobs[7]["test_end_date"], "2025-11-28")
        self.assertEqual(config, {"test_start_date": "2025-01-01", "run_name": "base"})

    def test_aggregates_months_separately_and_records_missing_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = [
                {"name": "may_1", "month": "may", "test_start_date": "2025-05-01", "test_end_date": "2025-05-07"},
                {"name": "may_2", "month": "may", "test_start_date": "2025-05-08", "test_end_date": "2025-05-14"},
                {"name": "nov_1", "month": "november", "test_start_date": "2025-11-01", "test_end_date": "2025-11-07"},
                {"name": "nov_missing", "month": "november", "test_start_date": "2025-11-08", "test_end_date": "2025-11-14"},
            ]
            (output / "manifest.yaml").write_text(yaml.safe_dump(manifest))
            for name, metrics in (
                ("may_1", {"mae": 1.0, "corr": None}),
                ("may_2", {"mae": 3.0, "corr": 0.5}),
                ("nov_1", {"mae": 5.0, "corr": 0.8}),
            ):
                run = output / "runs" / name / "result"
                run.mkdir(parents=True)
                (run / "metrics.yaml").write_text(yaml.safe_dump(metrics))

            destination = aggregate_metrics(output)
            summary = yaml.safe_load(destination.read_text())
            self.assertEqual(summary["monthly_metrics"]["may"]["mae"]["mean"], 2.0)
            self.assertEqual(summary["monthly_metrics"]["may"]["corr"]["n_windows"], 1)
            self.assertEqual(summary["monthly_metrics"]["november"]["mae"]["mean"], 5.0)
            self.assertEqual(summary["missing_windows"], ["nov_missing"])
            self.assertTrue((output / "per_window_metrics.csv").is_file())
            self.assertTrue((output / "monthly_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
