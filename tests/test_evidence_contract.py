"""Regression tests for the frozen NeuroChip evidence contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from src.evaluate_experiment_portfolio import greedy, objective, prepare_candidates
from src.audit_portfolio_one_swap import audit_selection


ROOT = Path(__file__).resolve().parents[1]


class EvidenceContractTests(unittest.TestCase):
    def require_reproduced_files(self, *relative_paths: str) -> None:
        """Skip evidence checks until their data-derived artifacts are reproduced."""
        missing = [str(path) for path in relative_paths if not (ROOT / path).is_file()]
        if missing:
            self.skipTest(
                "reproduction artifacts are not present in this source-only checkout; "
                "run the data and analysis steps in DATA_ACCESS_AND_REPRODUCTION.md first. "
                f"Missing: {', '.join(missing)}"
            )

    def test_dishbrain_experiment_contract(self):
        self.require_reproduced_files(
            "data/processed/dishbrain/drug_experiment_table.csv",
            "data/processed/dishbrain/dishbrain_audit.json",
        )
        table = pd.read_csv(ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv")
        audit = json.loads(
            (ROOT / "data" / "processed" / "dishbrain" / "dishbrain_audit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(table), 112)
        self.assertEqual(table["chip_id"].nunique(), 30)
        self.assertTrue(audit["first_rest_precedes_active"])
        self.assertTrue(audit["all_experiments_have_active_outcome"])

    def test_oof_prediction_contract(self):
        self.require_reproduced_files(
            "results/dishbrain_fingerprint_pilot/grouped_oof_predictions.csv"
        )
        predictions = pd.read_csv(
            ROOT / "results" / "dishbrain_fingerprint_pilot" / "grouped_oof_predictions.csv"
        )
        best = predictions.loc[
            predictions["feature_set"].eq("full_rest_electrophysiology")
            & predictions["model"].eq("extra_trees")
        ]
        self.assertEqual(len(best), 109)
        self.assertEqual(best["row_index"].nunique(), 109)
        self.assertTrue(best["prediction"].isin(best["truth"].unique()).all())
        probability_sum = best[[column for column in best if column.startswith("p_")]].sum(axis=1)
        self.assertTrue((probability_sum.sub(1).abs() < 1e-9).all())

    def test_claim_boundary_contract(self):
        self.require_reproduced_files("results/protocol_confound_scan/confound_scan.json")
        audit = json.loads(
            (ROOT / "results" / "protocol_confound_scan" / "confound_scan.json").read_text(encoding="utf-8")
        )
        self.assertEqual(audit["risk"], "HIGH")
        self.assertGreater(
            audit["metadata_negative_controls"]["within_chip_protocol_order"]["balanced_accuracy"],
            0.9,
        )
        self.assertIn("causal identification of drug mechanism", audit["blocked_claims"])

    def test_shared_chip_protocol_block_order_audit(self):
        self.require_reproduced_files(
            "results/dishbrain_fingerprint_robustness/robustness_audit.json",
            "results/dishbrain_fingerprint_robustness/shared_chip_protocol_block_order.csv",
        )
        audit = json.loads(
            (ROOT / "results" / "dishbrain_fingerprint_robustness" / "robustness_audit.json")
            .read_text(encoding="utf-8")
        )
        blocks = audit["shared_chip_protocol_block_order"]
        table = pd.read_csv(
            ROOT / "results" / "dishbrain_fingerprint_robustness" / "shared_chip_protocol_block_order.csv"
        )
        self.assertEqual(blocks["n_shared_chips"], 11)
        self.assertEqual(blocks["chips_following_modal_first_appearance_order"], 11)
        self.assertEqual(blocks["modal_first_appearance_order"], "Phenytoin > Perampanel > Carbamazepine")
        self.assertEqual(blocks["chips_with_three_contiguous_treatment_blocks"], 11)
        self.assertEqual(len(table), 11)
        self.assertTrue(table["each_drug_in_one_contiguous_block"].all())

    def test_headline_metric_contract(self):
        self.require_reproduced_files("results/dishbrain_fingerprint_pilot/grouped_cv_metrics.csv")
        metrics = pd.read_csv(
            ROOT / "results" / "dishbrain_fingerprint_pilot" / "grouped_cv_metrics.csv"
        )
        row = metrics.loc[
            metrics["feature_set"].eq("full_rest_electrophysiology")
            & metrics["model"].eq("extra_trees")
        ].iloc[0]
        self.assertAlmostEqual(row["balanced_accuracy"], 0.7374772313296902, places=10)
        self.assertAlmostEqual(row["macro_f1"], 0.7155083464984952, places=10)

    def test_posthoc_dose_negative_control_contract(self):
        self.require_reproduced_files(
            "results/dishbrain_dose_negative_control/dose_negative_control_summary.json"
        )
        report = json.loads(
            (ROOT / "results" / "dishbrain_dose_negative_control" / "dose_negative_control_summary.json")
            .read_text(encoding="utf-8")
        )
        full, shared = report["results"]
        self.assertEqual(full["records"], 109)
        self.assertEqual(full["chips"], 29)
        self.assertAlmostEqual(full["balanced_accuracy"], 0.7841530054644809, places=10)
        self.assertEqual(shared["records"], 64)
        self.assertEqual(shared["chips"], 11)
        self.assertAlmostEqual(shared["balanced_accuracy"], 0.8166666666666668, places=10)
        self.assertIn("post-hoc", report["purpose"])
        self.assertIn("does not identify a unique causal source", report["interpretation"])

    def test_redteam_dashboard_metrics_are_traceable(self):
        self.require_reproduced_files(
            "results/dishbrain_fingerprint_robustness/robustness_audit.json",
            "results/dishbrain_dose_negative_control/dose_negative_control_summary.json",
            "results/protocol_confound_scan/confound_scan.json",
        )
        robustness = json.loads(
            (ROOT / "results/dishbrain_fingerprint_robustness" / "robustness_audit.json")
            .read_text(encoding="utf-8")
        )
        dose_report = json.loads(
            (ROOT / "results/dishbrain_dose_negative_control" / "dose_negative_control_summary.json")
            .read_text(encoding="utf-8")
        )
        confound = json.loads(
            (ROOT / "results/protocol_confound_scan" / "confound_scan.json")
            .read_text(encoding="utf-8")
        )
        ephys = robustness["shared_chip_subset"]
        dose = next(
            item for item in dose_report["results"]
            if item["scope"] == "three_drug_shared_chip_subset"
        )
        order = confound["metadata_negative_controls"]["within_chip_protocol_order"]

        self.assertEqual((ephys["rows"], ephys["chips"]), (64, 11))
        self.assertEqual((dose["records"], dose["chips"]), (64, 11))
        self.assertEqual((order["n"], order["groups"]), (64, 11))
        self.assertAlmostEqual(ephys["balanced_accuracy"], 0.6893939393939393, places=12)
        self.assertAlmostEqual(dose["balanced_accuracy"], 0.8166666666666668, places=12)
        self.assertAlmostEqual(order["balanced_accuracy"], 0.9681818181818181, places=12)
        self.assertIn("post-hoc", dose_report["purpose"])
        self.assertIn("causal", dose_report["interpretation"])
        self.assertEqual(confound["risk"], "HIGH")

    def test_qubo_matches_portfolio_objective(self):
        self.require_reproduced_files(
            "results/experiment_portfolio/selected_candidates.csv",
            "data/processed/dishbrain/drug_experiment_table.csv",
            "results/dishbrain_fingerprint_pilot/grouped_oof_predictions.csv",
            "results/experiment_portfolio/portfolio_qubo_k5.npz",
            "results/experiment_portfolio/portfolio_qubo_k8.npz",
            "results/experiment_portfolio/portfolio_qubo_k12.npz",
        )
        selections = pd.read_csv(ROOT / "results" / "experiment_portfolio" / "selected_candidates.csv")
        candidates = pd.read_csv(ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv")
        candidates = candidates.loc[
            candidates["drug"].isin(["Carbamazepine", "Perampanel", "Phenytoin"])
        ].reset_index(drop=True)
        predictions = pd.read_csv(
            ROOT / "results" / "dishbrain_fingerprint_pilot" / "grouped_oof_predictions.csv"
        )
        predictions = predictions.loc[
            predictions["feature_set"].eq("full_rest_electrophysiology")
            & predictions["model"].eq("extra_trees")
        ].sort_values("row_index")
        probability = predictions[[column for column in predictions if column.startswith("p_")]].to_numpy()
        uncertainty = -(probability * np.log(probability + 1e-12)).sum(axis=1) / np.log(3)
        # Load the exact frozen diversity geometry via the production function.
        from src.evaluate_experiment_portfolio import prepare_candidates

        _, frozen_uncertainty, distance = prepare_candidates()
        self.assertTrue(np.allclose(uncertainty, frozen_uncertainty))
        for budget in [5, 8, 12]:
            selected = selections.loc[
                selections["selection"].eq(f"k{budget}_greedy_hybrid"), "candidate_index"
            ].astype(int).tolist()
            q = np.load(ROOT / "results" / "experiment_portfolio" / f"portfolio_qubo_k{budget}.npz")["Q"]
            x = np.zeros(len(candidates))
            x[selected] = 1
            score = objective(selected, uncertainty, distance, 0.55)["objective"]
            # The QUBO omits the constant A*k^2.
            recovered_score = -(float(x @ q @ x) + 2.0 * budget * budget)
            self.assertAlmostEqual(score, recovered_score, places=10)

    def test_greedy_portfolios_are_one_swap_local_optima(self):
        self.require_reproduced_files(
            "data/processed/dishbrain/drug_experiment_table.csv",
            "results/dishbrain_fingerprint_pilot/grouped_oof_predictions.csv",
        )
        _, uncertainty, distance = prepare_candidates()
        self.assertEqual(len(uncertainty), 109)
        for budget in [5, 8, 12]:
            selected = greedy(budget, uncertainty, distance, 0.55)
            audit = audit_selection(selected, uncertainty, distance, 0.55)
            self.assertEqual(audit["neighbor_count_evaluated"], budget * (109 - budget))
            self.assertEqual(audit["strictly_improving_neighbors_at_1e-12"], 0)
            self.assertTrue(audit["is_one_swap_local_optimum"])

    def test_random_portfolio_replay_audit(self):
        audit_dir = ROOT / "results" / "portfolio_random_audit_20260927"
        self.require_reproduced_files(
            "results/portfolio_random_audit_20260927/audit_manifest.json",
            "results/portfolio_random_audit_20260927/random_comparison_summary.csv",
            "results/portfolio_random_audit_20260927/random_portfolio_draws.csv",
        )
        manifest = json.loads((audit_dir / "audit_manifest.json").read_text(encoding="utf-8"))
        summary = pd.read_csv(audit_dir / "random_comparison_summary.csv")
        draws = pd.read_csv(audit_dir / "random_portfolio_draws.csv")
        self.assertTrue(manifest["replay_percentiles_match_frozen_results"])
        self.assertEqual(manifest["random_repeats_per_budget"], 5000)
        self.assertEqual(len(summary), 3)
        self.assertEqual(len(draws), 15000)
        self.assertTrue(summary["greedy_at_or_above_all_random_draws"].all())


if __name__ == "__main__":
    unittest.main()
