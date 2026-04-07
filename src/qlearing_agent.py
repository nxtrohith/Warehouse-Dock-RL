"""Q-Learning agent for the warehouse dock scheduling environment."""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

from src.config import NUM_DOCKS, VALID_ACTIONS


class StateEncoder:
    """Encodes continuous observations into discrete state indices."""

    def __init__(self) -> None:
        self.waiting_trucks_bins = 10
        self.dock_status_bins = 2 ** NUM_DOCKS
        self.time_remaining_bins = 8

    def encode(self, obs: Any) -> int:
        """Convert an observation model or dict to a discrete state index."""
        obs_data = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
        waiting_trucks = min(int(obs_data["waiting_trucks"]), self.waiting_trucks_bins - 1)
        dock_status_tuple = tuple(obs_data["dock_status"])
        dock_status_idx = self._binary_tuple_to_int(dock_status_tuple)
        time_remaining = min(int(obs_data["time_remaining"]), self.time_remaining_bins - 1)

        state_idx = (
            waiting_trucks * self.dock_status_bins * self.time_remaining_bins
            + dock_status_idx * self.time_remaining_bins
            + time_remaining
        )
        return state_idx

    def get_state_space_size(self) -> int:
        """Total number of discrete states."""
        return self.waiting_trucks_bins * self.dock_status_bins * self.time_remaining_bins

    @staticmethod
    def _binary_tuple_to_int(binary_tuple: Tuple[int, ...]) -> int:
        """Convert a tuple of binary values to an integer."""
        result = 0
        for bit in binary_tuple:
            result = (result << 1) | bit
        return result


class QLearningAgent:
    """Q-Learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        state_encoder: StateEncoder,
        num_actions: int = len(VALID_ACTIONS),
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        self.state_encoder = state_encoder
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        state_space_size = state_encoder.get_state_space_size()
        self.q_table: Dict[Tuple[int, int], float] = {}
        self._init_q_table(state_space_size, num_actions)

    def _init_q_table(self, state_space_size: int, num_actions: int) -> None:
        """Initialize Q-table with zeros."""
        for s in range(state_space_size):
            for a in range(num_actions):
                self.q_table[(s, a)] = 0.0

    def select_action(self, obs: Any, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        state = self.state_encoder.encode(obs)

        if training and random.random() < self.epsilon:
            return random.choice(range(self.num_actions))

        q_values = [self.q_table[(state, a)] for a in range(self.num_actions)]
        max_q = max(q_values)
        best_actions = [a for a in range(self.num_actions) if self.q_table[(state, a)] == max_q]
        return random.choice(best_actions)

    def update(
        self,
        obs: Any,
        action: int,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Q-learning update rule."""
        state = self.state_encoder.encode(obs)
        next_state = self.state_encoder.encode(next_obs)

        current_q = self.q_table[(state, action)]
        max_next_q = max(self.q_table[(next_state, a)] for a in range(self.num_actions))

        if done:
            max_next_q = 0.0

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, obs: Any) -> Dict[int, float]:
        """Get Q-values for all actions in current state."""
        state = self.state_encoder.encode(obs)
        return {a: self.q_table[(state, a)] for a in range(self.num_actions)}
