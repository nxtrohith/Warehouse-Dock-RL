"""Basic behavior checks for the warehouse dock environment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ACTION_HOLD, NUM_DOCKS
from src.env import WarehouseDockEnv


class TestWarehouseDockEnv(unittest.TestCase):
    def test_reset_returns_expected_keys(self) -> None:
        env = WarehouseDockEnv(seed=1)
        obs = env.reset()
        self.assertGreaterEqual(obs.waiting_trucks, 0)
        self.assertIsInstance(obs.queue_unload_times, list)
        self.assertIsInstance(obs.dock_status, list)
        self.assertIsInstance(obs.unloading_times, list)
        self.assertGreaterEqual(obs.time_remaining, 0)

    def test_step_returns_expected_types(self) -> None:
        env = WarehouseDockEnv(seed=2)
        env.reset()
        obs, reward, done, info = env.step(ACTION_HOLD)
        self.assertTrue(hasattr(obs, "waiting_trucks"))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_state_updates_after_step(self) -> None:
        env = WarehouseDockEnv(seed=3)
        env.reset()
        before = env.state()
        env.step(ACTION_HOLD)
        after = env.state()
        self.assertEqual(before.current_step + 1, after.current_step)

    def test_assign_action_maps_to_dock_index(self) -> None:
        env = WarehouseDockEnv(seed=5, enable_arrivals=False)
        env.reset()
        obs, _, _, info = env.step(1)
        self.assertEqual(info["action_meaning"], "assign_front_truck_to_dock_0")
        self.assertEqual(len(obs.dock_status), NUM_DOCKS)

    def test_episode_terminates_at_max_steps(self) -> None:
        env = WarehouseDockEnv(seed=4, max_steps=5)
        env.reset()

        done = False
        while not done:
            _, _, done, _ = env.step(ACTION_HOLD)

        self.assertTrue(done)
        self.assertEqual(env.state().current_step, 5)


if __name__ == "__main__":
    unittest.main()
