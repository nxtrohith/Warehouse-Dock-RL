"""Compare Q-Learning agent performance against random baseline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import VALID_ACTIONS
from src.env import WarehouseDockEnv
from src.qlearning_agent import QLearningAgent, StateEncoder


def run_random_baseline(num_episodes: int = 10, max_steps: int = 32) -> float:
    """Run environment with random actions."""
    import random

    env = WarehouseDockEnv(seed=7, max_steps=max_steps, enable_arrivals=False)
    total_rewards = []

    for _ in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = random.choice(VALID_ACTIONS)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)

    return sum(total_rewards) / len(total_rewards)


def run_qlearning_agent(num_episodes: int = 10, max_steps: int = 32) -> float:
    """Run environment with trained Q-Learning agent."""
    from src.qlearning_agent import QLearningAgent, StateEncoder

    env = WarehouseDockEnv(seed=7, max_steps=max_steps, enable_arrivals=False)
    state_encoder = StateEncoder()
    agent = QLearningAgent(state_encoder=state_encoder, epsilon=0.05)

    # Quick training
    for _ in range(50):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
        agent.decay_epsilon()

    # Evaluate
    env = WarehouseDockEnv(seed=7, max_steps=max_steps, enable_arrivals=False)
    total_rewards = []

    for _ in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)

    return sum(total_rewards) / len(total_rewards)


def main() -> None:
    print("=" * 60)
    print("Q-Learning vs Random Baseline Comparison")
    print("=" * 60)

    print("\nRunning random baseline (10 episodes)...")
    random_avg = run_random_baseline(num_episodes=10)
    print(f"Random baseline average reward: {random_avg:.2f}")

    print("\nTraining and evaluating Q-Learning agent...")
    ql_avg = run_qlearning_agent(num_episodes=10)
    print(f"Q-Learning average reward: {ql_avg:.2f}")

    improvement = ql_avg - random_avg
    improvement_pct = (improvement / abs(random_avg)) * 100 if random_avg != 0 else 0
    print(f"\nImprovement: +{improvement:.2f} ({improvement_pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
