"""UI-level regression check for the red-team evidence display."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class RedteamDashboardUITests(unittest.TestCase):
    def test_overview_auc_matches_frozen_grouped_cv_metrics(self):
        required_inputs = (
            "data/processed/dishbrain/drug_experiment_table.csv",
            "results/dishbrain_fingerprint_pilot/grouped_oof_predictions.csv",
            "results/dishbrain_fingerprint_pilot/grouped_cv_metrics.csv",
            "results/dishbrain_fingerprint_pilot/confusion_full_rest_electrophysiology_extra_trees.csv",
            "results/dishbrain_fingerprint_robustness/robustness_audit.json",
            "results/dishbrain_fingerprint_robustness/cluster_bootstrap.csv",
            "results/dishbrain_dose_negative_control/dose_negative_control_summary.json",
            "results/protocol_confound_scan/confound_scan.json",
        )
        missing = [path for path in required_inputs if not (ROOT / path).is_file()]
        if missing:
            self.skipTest(
                "overview UI check needs locally reproduced analysis artifacts; "
                "follow DATA_ACCESS_AND_REPRODUCTION.md first. Missing: " + ", ".join(missing)
            )

        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)

        self.assertFalse(app.exception)
        metrics = {metric.label: (metric.value, metric.delta) for metric in app.metric}
        self.assertEqual(metrics["宏 OVR AUC"], ("0.8794", ""))

    def test_source_only_checkout_shows_setup_instructions_instead_of_traceback(self):
        required_inputs = (
            "data/processed/dishbrain/drug_experiment_table.csv",
            "data/processed/dishbrain/dishbrain_audit.json",
            "data/processed/dataset_audit.json",
            "results/dishbrain_fingerprint_pilot/grouped_oof_predictions.csv",
            "results/dishbrain_fingerprint_pilot/grouped_cv_metrics.csv",
            "results/dishbrain_fingerprint_pilot/confusion_full_rest_electrophysiology_extra_trees.csv",
            "results/dishbrain_fingerprint_robustness/robustness_audit.json",
            "results/dishbrain_fingerprint_robustness/shared_chip_protocol_block_order.csv",
            "results/dishbrain_fingerprint_robustness/cluster_bootstrap.csv",
            "results/dishbrain_dose_negative_control/dose_negative_control_summary.json",
            "results/protocol_confound_scan/confound_scan.json",
            "results/organoid_recording_qc/recording_qc.csv",
            "results/organoid_recording_qc/recording_qc_summary.json",
        )
        missing = [path for path in required_inputs if not (ROOT / path).is_file()]
        if not missing:
            self.skipTest("source-only startup check applies to the data-free release checkout")

        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)

        self.assertFalse(app.exception)
        self.assertTrue(any("完整演示尚未就绪" in item.value for item in app.warning))
        self.assertTrue(any("DATA_ACCESS_AND_REPRODUCTION.md" in item.value for item in app.markdown))
        self.assertTrue(any("没有附带" in item.value for item in app.warning))

    def test_project_provenance_copy_focuses_on_this_challenge(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("本原型基于团队此前开发的神经微生理系统分析项目", app_source)
        self.assertIn("本届聚焦 MEA 记录审阅与流程风险核查", app_source)
        self.assertNotIn("噬菌体 Ising 量子赛", app_source)
        self.assertNotIn("原型承接此前量子计算竞赛", app_source)

    def test_cached_data_key_is_bumped_for_corrected_earliest_rest_metrics(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('cache_version="earliest-rest-row-fix-20261003"', app_source)

    def test_organoid_qc_summary_metrics_are_compact_and_correct(self):
        # The app loads DishBrain, robustness, and negative-control artifacts
        # before rendering any page. A source-only release intentionally omits
        # those data-derived files, so this integration check runs only after
        # the documented local replay has recreated its inputs.
        required_inputs = (
            "data/processed/dishbrain/drug_experiment_table.csv",
            "results/dishbrain_fingerprint_pilot/grouped_oof_predictions.csv",
            "results/dishbrain_fingerprint_pilot/grouped_cv_metrics.csv",
            "results/dishbrain_fingerprint_pilot/confusion_full_rest_electrophysiology_extra_trees.csv",
            "results/dishbrain_fingerprint_robustness/robustness_audit.json",
            "results/dishbrain_fingerprint_robustness/cluster_bootstrap.csv",
            "results/dishbrain_dose_negative_control/dose_negative_control_summary.json",
            "results/protocol_confound_scan/confound_scan.json",
            "results/organoid_recording_qc/recording_qc.csv",
            "results/organoid_recording_qc/recording_qc_summary.json",
            "data/processed/dataset_audit.json",
        )
        missing = [path for path in required_inputs if not (ROOT / path).is_file()]
        if missing:
            self.skipTest(
                "UI integration check needs locally reproduced data-derived artifacts; "
                "follow DATA_ACCESS_AND_REPRODUCTION.md first. Missing: " + ", ".join(missing)
            )

        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
        app.radio[0].set_value("类器官记录质控").run(timeout=20)

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["类器官记录"], "45")
        self.assertEqual(metrics["时间戳结构通过"], "45/45")
        self.assertEqual(metrics["计数不匹配记录"], "1 份")

    def test_exact_metrics_and_denominators_are_visible(self):
        # The public/source-only package intentionally omits rights-sensitive
        # data-derived tables. Match the repository's evidence-contract tests:
        # skip this UI integration check until the documented local replay has
        # recreated the required inputs, rather than failing at app startup.
        required_inputs = (
            "data/processed/dishbrain/drug_experiment_table.csv",
            "data/processed/dishbrain/dishbrain_audit.json",
            "results/dishbrain_fingerprint_pilot/grouped_oof_predictions.csv",
            "results/dishbrain_fingerprint_pilot/grouped_cv_metrics.csv",
            "results/dishbrain_fingerprint_pilot/confusion_full_rest_electrophysiology_extra_trees.csv",
            "results/dishbrain_fingerprint_robustness/robustness_audit.json",
            "results/dishbrain_fingerprint_robustness/cluster_bootstrap.csv",
            "results/dishbrain_dose_negative_control/dose_negative_control_summary.json",
            "results/protocol_confound_scan/confound_scan.json",
        )
        missing = [path for path in required_inputs if not (ROOT / path).is_file()]
        if missing:
            self.skipTest(
                "UI integration check needs locally reproduced, rights-sensitive analysis artifacts; "
                "follow DATA_ACCESS_AND_REPRODUCTION.md first. Missing: " + ", ".join(missing)
            )

        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
        app.radio[0].set_value("可信度红队").run(timeout=20)

        self.assertFalse(app.exception)
        metrics = {metric.label: (metric.value, metric.delta) for metric in app.metric}
        self.assertEqual(
            metrics["电生理模型 · 共享子集"],
            ("68.94%", ""),
        )
        self.assertEqual(
            metrics["仅协议顺序 · 负对照"],
            ("96.82%", ""),
        )
        self.assertEqual(
            metrics["仅剂量 · 事后负对照"],
            ("81.67%", ""),
        )
        self.assertTrue(any("64 条记录 / 11 个芯片" in block.value for block in app.caption))
        self.assertEqual(metrics["遵循共同顺序"], ("11/11", ""))
        self.assertEqual(metrics["三个连续处理区块"], ("11/11", ""))
        self.assertTrue(
            any(
                "Phenytoin → Perampanel → Carbamazepine" in block.value
                for block in app.markdown
            )
        )
        self.assertTrue(
            any("无法彼此拆开解释" in block.value for block in app.info)
        )
        self.assertTrue(
            any("不能确定因果来源" in block.value for block in app.markdown)
        )

    def test_planning_page_labels_retrospective_scope(self):
        required_inputs = (
            "data/processed/dishbrain/drug_experiment_table.csv",
            "results/dishbrain_fingerprint_pilot/grouped_oof_predictions.csv",
        )
        missing = [path for path in required_inputs if not (ROOT / path).is_file()]
        if missing:
            self.skipTest(
                "planning-page UI check requires locally reproduced data; "
                "follow DATA_ACCESS_AND_REPRODUCTION.md first. Missing: " + ", ".join(missing)
            )

        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
        app.radio[0].set_value("后续实验规划").run(timeout=20)

        self.assertFalse(app.exception)
        self.assertTrue(
            any("回顾性演示" in block.value and "未验证真实前瞻收益" in block.value for block in app.warning)
        )
        self.assertTrue(
            any("历史候选回放" in block.value and "供研究者审阅" in block.value for block in app.success)
        )


if __name__ == "__main__":
    unittest.main()
