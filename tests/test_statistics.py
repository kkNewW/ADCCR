from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paired_seed_stats import (  # noqa: E402
    holm_adjust,
    pairwise_ci,
    population_mean_sd,
    sample_sd,
)


class SeedStatisticsTest(unittest.TestCase):
    def test_population_sd_matches_table_14(self):
        mean, sd = population_mean_sd([78.3, 77.5, 77.0, 78.4, 77.8])
        self.assertAlmostEqual(mean, 77.8)
        self.assertAlmostEqual(sd, 0.5176871642217922)
        self.assertEqual(round(sd, 3), 0.518)

    def test_inferential_sd_remains_sample_based(self):
        values = [1.3, 0.8, -0.4, 1.2, 1.0]
        _, population_sd = population_mean_sd(values)
        self.assertGreater(sample_sd(values), population_sd)
        mean, low, high = pairwise_ci(values)
        self.assertAlmostEqual(mean, 0.78)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)

    def test_holm_adjustment_remains_available_for_optional_analysis(self):
        pairs = [{"p_exact": 0.01}, {"p_exact": 0.04}, {"p_exact": 0.03}]
        holm_adjust(pairs)
        self.assertEqual(
            [pair["p_holm"] for pair in pairs],
            [0.03, 0.06, 0.06],
        )


if __name__ == "__main__":
    unittest.main()
