"""Data-independent regression checks for the portfolio objective and QUBO."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from src.evaluate_experiment_portfolio import greedy, objective, qubo_matrix


class PortfolioMathTests(unittest.TestCase):
    def setUp(self) -> None:
        # Small deterministic fixture; these values are synthetic test inputs,
        # not biological observations and not part of reported results.
        self.uncertainty = np.array([0.15, 0.82, 0.47, 0.63], dtype=float)
        self.distance = np.array(
            [
                [0.0, 0.20, 0.75, 0.45],
                [0.20, 0.0, 0.55, 0.90],
                [0.75, 0.55, 0.0, 0.35],
                [0.45, 0.90, 0.35, 0.0],
            ],
            dtype=float,
        )
        self.weight = 0.55

    def test_qubo_energy_matches_objective_for_every_fixed_size_subset(self) -> None:
        penalty = 2.3
        for budget in (2, 3):
            q = qubo_matrix(budget, self.uncertainty, self.distance, self.weight, penalty)
            # For fixed-cardinality states, x'Qx differs from -objective only
            # by the known constant -penalty * budget**2.
            for selected in itertools.combinations(range(len(self.uncertainty)), budget):
                binary = np.zeros(len(self.uncertainty), dtype=float)
                binary[list(selected)] = 1.0
                direct = objective(list(selected), self.uncertainty, self.distance, self.weight)["objective"]
                mapped = -(float(binary @ q @ binary) + penalty * budget**2)
                self.assertAlmostEqual(direct, mapped, places=12)

    def test_greedy_returns_a_unique_in_range_fixed_size_selection(self) -> None:
        selected = greedy(3, self.uncertainty, self.distance, self.weight)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        self.assertTrue(all(0 <= index < len(self.uncertainty) for index in selected))
        self.assertTrue(np.isfinite(objective(selected, self.uncertainty, self.distance, self.weight)["objective"]))


if __name__ == "__main__":
    unittest.main()
