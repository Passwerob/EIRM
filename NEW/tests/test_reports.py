from __future__ import annotations

import unittest

from ami_active_sensing.evaluation.reports import select_winners, winner_distribution


def record(name: str, recall: float, count: int, density: float) -> dict[str, object]:
    return {
        "policy": name,
        "aggregate": {
            "mean_repeatability": recall,
            "mean_matching_precision": recall,
            "mean_matching_recall": recall,
            "mean_ransac_inlier_ratio": recall,
            "matching_failure_rate": 0.0,
            "mean_event_density": density,
            "event_count": count,
        },
    }


class ReportTests(unittest.TestCase):
    def test_count_and_task_winners_are_independent(self) -> None:
        records = [record("useful", 0.9, 100, 1.0), record("many_events", 0.2, 1000, 8.0)]
        winners = select_winners(records, bandwidth_limit=10.0, bandwidth_penalty=0.01)
        self.assertEqual(winners["best_task"], "useful")
        self.assertEqual(winners["max_event_count"], "many_events")
        self.assertGreater(winners["count_objective_task_regret"], 0.0)

    def test_hard_budget_excludes_task_winner(self) -> None:
        records = [record("over_budget", 0.9, 1000, 12.0), record("feasible", 0.7, 500, 5.0)]
        winners = select_winners(records, bandwidth_limit=10.0, bandwidth_penalty=0.0)
        self.assertEqual(winners["best_task"], "over_budget")
        self.assertEqual(winners["best_useful_under_budget"], "feasible")
        self.assertTrue(winners["budget_feasible"])

    def test_infeasible_budget_is_explicit(self) -> None:
        records = [record("lower", 0.4, 100, 4.0), record("higher", 0.8, 200, 8.0)]
        winners = select_winners(records, bandwidth_limit=2.0, bandwidth_penalty=0.0)
        self.assertFalse(winners["budget_feasible"])
        self.assertEqual(winners["best_useful_under_budget"], "lower")

    def test_winner_distribution_reports_heterogeneity(self) -> None:
        summary = winner_distribution(["a", "a", "b", "c"])
        self.assertEqual(summary["unique_winner_count"], 3)
        self.assertFalse(summary["single_trajectory_wins_everywhere"])
        self.assertGreater(summary["normalized_winner_entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
