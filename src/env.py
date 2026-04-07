"""Minimal real-world RL environment: warehouse dock scheduling."""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

from src.config import (
    ARRIVAL_PROBABILITY,
    ACTION_HOLD,
    DONE_WHEN_ALL_PROCESSED,
    ENABLE_ARRIVALS,
    INITIAL_QUEUE_SIZE,
    MAX_STEPS,
    MAX_UNLOAD_TICKS,
    MIN_UNLOAD_TICKS,
    NUM_DOCKS,
    PENALTY_IDLE_DOCK_WHILE_QUEUE,
    PENALTY_HOLD_WHEN_ASSIGN_AVAILABLE,
    PENALTY_INVALID_ACTION,
    PENALTY_WAIT_PER_TRUCK,
    QUEUE_FEATURE_SIZE,
    REWARD_COMPLETION,
    REWARD_HOLD_NO_VALID_ASSIGN,
    REWARD_VALID_ASSIGN,
    VALID_ACTIONS,
)
from src.openenv_models import Observation, State, StepResponse


class WarehouseDockEnv:
    """Simple RL environment interface with reset, state, and step."""

    def __init__(
        self,
        seed: int = 7,
        max_steps: int = MAX_STEPS,
        enable_arrivals: bool = ENABLE_ARRIVALS,
        done_when_all_processed: bool = DONE_WHEN_ALL_PROCESSED,
    ) -> None:
        self._rng = random.Random(seed)
        self.max_steps = max_steps
        self.enable_arrivals = enable_arrivals
        self.done_when_all_processed = done_when_all_processed
        self.current_step = 0
        self.queue: List[int] = []
        self.unloading_times: List[int] = [0] * NUM_DOCKS
        self.processed_trucks = 0
        self.total_trucks_created = 0

    def reset(self) -> Observation:
        self.current_step = 0
        self.queue = [self._sample_unload_time() for _ in range(INITIAL_QUEUE_SIZE)]
        self.unloading_times = [0] * NUM_DOCKS
        self.processed_trucks = 0
        self.total_trucks_created = len(self.queue)
        return Observation(
            waiting_trucks=len(self.queue),
            queue_unload_times=self._queue_features(),
            dock_status=self._dock_status(),
            unloading_times=copy.deepcopy(self.unloading_times),
            time_remaining=self._time_remaining(),
        )

    def step(self, action: int):
        reward_parts = {
            "valid_assignment_reward": 0.0,
            "completion_reward": 0.0,
            "wait_penalty": 0.0,
            "idle_dock_penalty": 0.0,
            "invalid_action_penalty": 0.0,
            "hold_adjustment": 0.0,
        }

        valid_assignment_available = self._has_valid_assignment()

        if action not in VALID_ACTIONS:
            action = ACTION_HOLD
            reward_parts["invalid_action_penalty"] += PENALTY_INVALID_ACTION

        assigned = self._apply_action(action)
        if action == ACTION_HOLD:
            if valid_assignment_available:
                reward_parts["hold_adjustment"] += PENALTY_HOLD_WHEN_ASSIGN_AVAILABLE
            else:
                reward_parts["hold_adjustment"] += REWARD_HOLD_NO_VALID_ASSIGN
        elif assigned:
            reward_parts["valid_assignment_reward"] += REWARD_VALID_ASSIGN
        else:
            reward_parts["invalid_action_penalty"] += PENALTY_INVALID_ACTION

        completed = self._advance_one_tick()
        reward_parts["completion_reward"] += completed * REWARD_COMPLETION

        if self.enable_arrivals:
            self._spawn_new_arrivals()

        waiting_trucks = len(self.queue)
        reward_parts["wait_penalty"] += waiting_trucks * PENALTY_WAIT_PER_TRUCK

        idle_docks = sum(1 for remaining in self.unloading_times if remaining == 0)
        if waiting_trucks > 0 and idle_docks > 0:
            reward_parts["idle_dock_penalty"] += idle_docks * PENALTY_IDLE_DOCK_WHILE_QUEUE

        self.current_step += 1
        done = self.current_step >= self.max_steps
        if self.done_when_all_processed and self._all_processed():
            done = True

        reward = float(sum(reward_parts.values()))
        info = {
            "reward_parts": reward_parts,
            "assigned": assigned,
            "completed": completed,
            "action_meaning": self.action_meaning(action),
        }
        step_response = StepResponse(
            observation=self._observation(),
            reward=reward,
            done=done,
            info=info,
        )
        return (
            step_response.observation,
            step_response.reward,
            step_response.done,
            step_response.info,
        )

    def state(self) -> State:
        return State(
            current_step=self.current_step,
            waiting_trucks=len(self.queue),
            queue_unload_times=self._queue_features(),
            dock_status=self._dock_status(),
            unloading_times=copy.deepcopy(self.unloading_times),
            time_remaining=self._time_remaining(),
            processed_trucks=self.processed_trucks,
            total_trucks_created=self.total_trucks_created,
        )

    def _apply_action(self, action: int) -> bool:
        if action == ACTION_HOLD:
            return True

        dock_idx = action - 1
        if dock_idx < 0 or dock_idx >= NUM_DOCKS:
            return False
        if self.unloading_times[dock_idx] != 0:
            return False
        if not self.queue:
            return False

        unload_time = self.queue.pop(0)
        self.unloading_times[dock_idx] = unload_time
        return True

    def _advance_one_tick(self) -> int:
        completed = 0
        for idx, remaining in enumerate(self.unloading_times):
            if remaining > 0:
                self.unloading_times[idx] -= 1
                if self.unloading_times[idx] == 0:
                    completed += 1
        self.processed_trucks += completed
        return completed

    def _spawn_new_arrivals(self) -> None:
        # Lightweight stochastic arrivals for realism.
        if self._rng.random() < ARRIVAL_PROBABILITY:
            self.queue.append(self._sample_unload_time())
            self.total_trucks_created += 1

    def _sample_unload_time(self) -> int:
        return self._rng.randint(MIN_UNLOAD_TICKS, MAX_UNLOAD_TICKS)

    def _dock_status(self) -> List[int]:
        return [1 if remaining > 0 else 0 for remaining in self.unloading_times]

    def _queue_features(self) -> List[int]:
        features = self.queue[:QUEUE_FEATURE_SIZE]
        if len(features) < QUEUE_FEATURE_SIZE:
            features.extend([0] * (QUEUE_FEATURE_SIZE - len(features)))
        return features

    def _time_remaining(self) -> int:
        return max(0, self.max_steps - self.current_step)

    def _has_valid_assignment(self) -> bool:
        return bool(self.queue) and any(remaining == 0 for remaining in self.unloading_times)

    def _all_processed(self) -> bool:
        no_waiting = len(self.queue) == 0
        all_docks_idle = all(remaining == 0 for remaining in self.unloading_times)
        return no_waiting and all_docks_idle

    def action_meaning(self, action: int) -> str:
        if action == ACTION_HOLD:
            return "hold"
        dock_idx = action - 1
        if 0 <= dock_idx < NUM_DOCKS:
            return f"assign_front_truck_to_dock_{dock_idx}"
        return "invalid"

    def _observation(self) -> Dict[str, object]:
        return {
            "waiting_trucks": len(self.queue),
            "queue_unload_times": self._queue_features(),
            "dock_status": self._dock_status(),
            "unloading_times": copy.deepcopy(self.unloading_times),
            "time_remaining": self._time_remaining(),
        }
