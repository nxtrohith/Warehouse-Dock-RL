"""Training loop for Q-Learning agent on warehouse dock scheduling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import WarehouseDockEnv
from src.qlearing_agent import QLearningAgent, StateEncoder


def train(
    num_episodes: int = 100,
    max_steps: int = 32,
    verbose: bool = True,
) -> tuple[QLearningAgent, list[float]]:
    """Train Q-Learning agent on warehouse dock environment."""
    env = WarehouseDockEnv(seed=42, max_steps=max_steps, enable_arrivals=False)
    state_encoder = StateEncoder()
    agent = QLearningAgent(state_encoder=state_encoder)

    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            total_reward += reward
            obs = next_obs

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Reward (last 10): {avg_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    return agent, episode_rewards


def evaluate(agent: QLearningAgent, num_episodes: int = 10, max_steps: int = 32) -> float:
    """Evaluate trained agent without exploration."""
    env = WarehouseDockEnv(seed=99, max_steps=max_steps, enable_arrivals=False)
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

    avg_reward = sum(total_rewards) / len(total_rewards)
    return avg_reward


def main() -> None:
    print("Starting Q-Learning training...")
    agent, episode_rewards = train(num_episodes=100, verbose=True)

    print("\nTraining complete. Evaluating trained agent...")
    avg_eval_reward = evaluate(agent, num_episodes=5)
    print(f"Average evaluation reward: {avg_eval_reward:.2f}")

    print("\nTraining summary:")
    print(f"Average reward (last 10 episodes): {sum(episode_rewards[-10:]) / 10:.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Worst episode reward: {min(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
