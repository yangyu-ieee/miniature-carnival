"""Regression test for preserving missing values in the earliest rest row."""

from __future__ import annotations

import unittest

import pandas as pd

from src.audit_dishbrain import select_earliest_rest_rows


class EarliestRestSemanticsTests(unittest.TestCase):
    def test_later_non_null_value_does_not_backfill_earliest_row(self):
        sessions = pd.DataFrame(
            [
                {"chip_id": "14218", "date": "2022-04-27", "drug": "Phenytoin", "dose": 33,
                 "session_num": 0, "status": "rest", "rest_DCC": None},
                {"chip_id": "14218", "date": "2022-04-27", "drug": "Phenytoin", "dose": 33,
                 "session_num": 1, "status": "full game", "rest_DCC": None},
                {"chip_id": "14218", "date": "2022-04-27", "drug": "Phenytoin", "dose": 33,
                 "session_num": 2, "status": "rest", "rest_DCC": 0.546},
                {"chip_id": "14229", "date": "2022-04-29", "drug": "Perampanel", "dose": 10,
                 "session_num": 1, "status": "rest", "rest_DCC": 0.7},
                {"chip_id": "14229", "date": "2022-04-29", "drug": "Perampanel", "dose": 10,
                 "session_num": 3, "status": "rest", "rest_DCC": 0.9},
            ]
        )

        selected = select_earliest_rest_rows(sessions).sort_values("chip_id").reset_index(drop=True)

        self.assertEqual(len(selected), 2)
        self.assertEqual(int(selected.loc[0, "session_num"]), 0)
        self.assertTrue(pd.isna(selected.loc[0, "rest_DCC"]))
        self.assertEqual(float(selected.loc[1, "rest_DCC"]), 0.7)
        self.assertEqual(int(selected.loc[1, "session_num"]), 1)


if __name__ == "__main__":
    unittest.main()
