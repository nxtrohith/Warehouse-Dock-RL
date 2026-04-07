"""Deterministic graders for dock scheduling tasks.

Each grader returns a score in [0.0, 1.0].
The input is expected to be a dict of evaluation metrics.
"""

from __future__ import annotations

from typing import Any, Mapping


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _get_float(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_int(metrics: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = metrics.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def task_1_grader(result: Mapping[str, Any]) -> float:
    """Score basic dock assignment quality.

    Expected keys:
    - invalid_action_count
    - assigned_within_first_n_steps
    - completed_trucks
    - initial_queue_size
    - no_invalid_actions (optional)
    - early_assignment (optional)
    - completion_rate (optional)
    """
    invalid_action_count = _get_int(result, "invalid_action_count")
    initial_queue_size = max(1, _get_int(result, "initial_queue_size", 1))
    completed_trucks = _get_int(result, "completed_trucks")
    assigned_within_first_n_steps = _get_int(result, "assigned_within_first_n_steps")

    no_invalid = 1.0 if invalid_action_count == 0 else 0.0
    early_assignment = 1.0 if assigned_within_first_n_steps > 0 else 0.0
    completion_rate = completed_trucks / initial_queue_size

    score = 0.4 * no_invalid + 0.3 * early_assignment + 0.3 * completion_rate
    return _clamp_score(score)


def task_2_grader(result: Mapping[str, Any]) -> float:
    """Score queue reduction and dock utilization.

    Expected keys:
    - invalid_action_count
    - mean_waiting_trucks
    - idle_dock_steps
    - max_waiting_trucks
    - initial_queue_size
    """
    invalid_action_count = _get_int(result, "invalid_action_count")
    mean_waiting_trucks = _get_float(result, "mean_waiting_trucks")
    idle_dock_steps = _get_float(result, "idle_dock_steps")
    max_waiting_trucks = max(1.0, _get_float(result, "max_waiting_trucks", 1.0))
    initial_queue_size = max(1.0, _get_float(result, "initial_queue_size", 1.0))

    no_invalid = 1.0 if invalid_action_count == 0 else 0.0
    normalized_queue_reduction = 1.0 - min(1.0, mean_waiting_trucks / max_waiting_trucks)
    dock_utilization = 1.0 - min(1.0, idle_dock_steps / initial_queue_size)

    score = 0.35 * no_invalid + 0.35 * normalized_queue_reduction + 0.30 * dock_utilization
    return _clamp_score(score)


def task_3_grader(result: Mapping[str, Any]) -> float:
    """Score full automation under stochastic arrivals.

    Expected keys:
    - invalid_action_count
    - average_reward
    - baseline_reward
    - completed_trucks
    - total_trucks_created
    - queue_remaining
    """
    invalid_action_count = _get_int(result, "invalid_action_count")
    average_reward = _get_float(result, "average_reward")
    baseline_reward = _get_float(result, "baseline_reward", 0.0)
    completed_trucks = _get_float(result, "completed_trucks")
    total_trucks_created = max(1.0, _get_float(result, "total_trucks_created", 1.0))
    queue_remaining = _get_float(result, "queue_remaining")

    no_invalid = 1.0 if invalid_action_count == 0 else 0.0
    reward_quality = _clamp_score((average_reward - baseline_reward) / max(1.0, abs(baseline_reward) + 10.0))
    throughput = _clamp_score(completed_trucks / total_trucks_created)
    queue_clearance = 1.0 - _clamp_score(queue_remaining / total_trucks_created)

    score = 0.30 * no_invalid + 0.30 * reward_quality + 0.20 * throughput + 0.20 * queue_clearance
    return _clamp_score(score)
