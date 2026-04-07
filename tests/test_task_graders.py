"""Tests for deterministic dock scheduling graders."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task_graders import task_1_grader, task_2_grader, task_3_grader


class TestTaskGraders(unittest.TestCase):
    def test_task_1_grader_returns_valid_score(self) -> None:
        score = task_1_grader(
            {
                "invalid_action_count": 0,
                "assigned_within_first_n_steps": 1,
                "completed_trucks": 5,
                "initial_queue_size": 5,
            }
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertAlmostEqual(score, 1.0)

    def test_task_2_grader_penalizes_bad_queue_behavior(self) -> None:
        score = task_2_grader(
            {
                "invalid_action_count": 2,
                "mean_waiting_trucks": 5.0,
                "idle_dock_steps": 6,
                "max_waiting_trucks": 10,
                "initial_queue_size": 5,
            }
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertLess(score, 0.5)

    def test_task_3_grader_handles_stochastic_metrics(self) -> None:
        score = task_3_grader(
            {
                "invalid_action_count": 0,
                "average_reward": 24.0,
                "baseline_reward": 8.0,
                "completed_trucks": 8,
                "total_trucks_created": 10,
                "queue_remaining": 1,
            }
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()
