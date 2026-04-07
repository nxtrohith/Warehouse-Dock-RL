"""Tests for Q-Learning agent."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import WarehouseDockEnv
from src.qlearing_agent import QLearningAgent, StateEncoder


class TestStateEncoder(unittest.TestCase):
    def test_encode_returns_valid_state_index(self) -> None:
        encoder = StateEncoder()
        env = WarehouseDockEnv(seed=1)
        obs = env.reset()

        state_idx = encoder.encode(obs)
        self.assertIsInstance(state_idx, int)
        self.assertGreaterEqual(state_idx, 0)
        self.assertLess(state_idx, encoder.get_state_space_size())

    def test_state_space_size_is_reasonable(self) -> None:
        encoder = StateEncoder()
        state_space_size = encoder.get_state_space_size()
        self.assertGreater(state_space_size, 0)
        self.assertLess(state_space_size, 100000)

    def test_same_observation_maps_to_same_state(self) -> None:
        encoder = StateEncoder()
        obs = {
            "waiting_trucks": 3,
            "queue_unload_times": [2, 3, 4, 0, 0],
            "dock_status": [1, 0, 1],
            "unloading_times": [2, 0, 1],
            "time_remaining": 10,
        }
        state1 = encoder.encode(obs)
        state2 = encoder.encode(obs)
        self.assertEqual(state1, state2)


class TestQLearningAgent(unittest.TestCase):
    def test_agent_initialization(self) -> None:
        encoder = StateEncoder()
        agent = QLearningAgent(state_encoder=encoder)
        self.assertIsNotNone(agent.q_table)
        self.assertEqual(len(agent.q_table), encoder.get_state_space_size() * 4)

    def test_agent_selects_valid_action(self) -> None:
        encoder = StateEncoder()
        agent = QLearningAgent(state_encoder=encoder)
        env = WarehouseDockEnv(seed=2)
        obs = env.reset()

        action = agent.select_action(obs, training=False)
        self.assertIn(action, range(4))

    def test_agent_update_changes_q_table(self) -> None:
        encoder = StateEncoder()
        agent = QLearningAgent(state_encoder=encoder)
        env = WarehouseDockEnv(seed=3)
        obs = env.reset()

        action = agent.select_action(obs)
        state_idx = encoder.encode(obs)
        q_before = agent.q_table[(state_idx, action)]

        next_obs, reward, done, _ = env.step(action)
        agent.update(obs, action, reward, next_obs, done)

        q_after = agent.q_table[(state_idx, action)]
        self.assertNotEqual(q_before, q_after)

    def test_epsilon_decay_decreases_exploration(self) -> None:
        encoder = StateEncoder()
        agent = QLearningAgent(state_encoder=encoder, epsilon=1.0)
        epsilon_start = agent.epsilon

        for _ in range(10):
            agent.decay_epsilon()

        epsilon_end = agent.epsilon
        self.assertLess(epsilon_end, epsilon_start)

    def test_get_q_values_returns_dict(self) -> None:
        encoder = StateEncoder()
        agent = QLearningAgent(state_encoder=encoder)
        env = WarehouseDockEnv(seed=4)
        obs = env.reset()

        q_values = agent.get_q_values(obs)
        self.assertIsInstance(q_values, dict)
        self.assertEqual(len(q_values), 4)


if __name__ == "__main__":
    unittest.main()
