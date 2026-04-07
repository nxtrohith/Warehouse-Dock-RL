"""Run a quick smoke test loop over the environment with random actions."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import VALID_ACTIONS
from src.env import WarehouseDockEnv


def main() -> None:
    rng = random.Random(11)
    env = WarehouseDockEnv(seed=42)
    obs = env.reset()

    total_reward = 0.0
    done = False
    steps = 0

    while not done:
        action = rng.choice(VALID_ACTIONS)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    print("Smoke run completed")
    print(f"steps={steps}")
    print(f"total_reward={total_reward:.2f}")
    print(f"final_obs={obs}")
    final_state = env.state()
    print(f"processed={final_state.processed_trucks}")
    print(f"waiting={final_state.waiting_trucks}")
    print(f"time_remaining={final_state.time_remaining}")


if __name__ == "__main__":
    main()
