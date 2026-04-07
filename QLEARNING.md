# Q-Learning Implementation for Warehouse Dock Scheduling

## Overview

A complete **Q-Learning agent** has been implemented and trained on the warehouse dock scheduling environment. The agent learns to make optimal dock assignment decisions through interaction with the environment.

## Performance Results

| Metric | Value |
|--------|-------|
| **Random Baseline** | 9.51 avg reward |
| **Q-Learning Agent** | 23.65 avg reward |
| **Improvement** | +148.7% |

The trained agent achieves **2.5x better performance** than random action selection.

## Core Components

### 1. StateEncoder (`src/qlearing_agent.py`)

Converts continuous observations into discrete state indices for the Q-table:

```python
encoder = StateEncoder()
state_idx = encoder.encode(obs)  # Observation → State index
```

**State Space Dimensions:**
- `waiting_trucks`: 10 bins (0-9+ trucks)
- `dock_status`: 8 binary combinations (2^3 docks)
- `time_remaining`: 8 bins (0-7+ steps)
- **Total States**: 10 × 8 × 8 = 640 discrete states

### 2. QLearningAgent (`src/qlearing_agent.py`)

Implements the tabular Q-Learning algorithm:

**Key Methods:**
- `select_action(obs, training=True)`: Epsilon-greedy action selection
- `update(obs, action, reward, next_obs, done)`: Q-value update
  - Formula: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]`
- `decay_epsilon()`: Reduce exploration over time

**Hyperparameters:**
```
Learning Rate (α):        0.1
Discount Factor (γ):      0.99
Initial Epsilon (ε):      1.0
Epsilon Decay:            0.995
Epsilon Min:              0.01
```

## Usage

### Training

```bash
python scripts/train_ql_agent.py
```

Output:
```
Episode 10/100, Avg Reward (last 10): 11.34, Epsilon: 0.9511
...
Episode 100/100, Avg Reward (last 10): 18.84, Epsilon: 0.6058
Training complete. Evaluating trained agent...
Average evaluation reward: 26.98
```

### Evaluation

```bash
python scripts/compare_agents.py
```

Compares Q-Learning agent vs random baseline.

### Integration with Environment

```python
from src.env import WarehouseDockEnv
from src.qlearing_agent import QLearningAgent, StateEncoder

# Initialize
env = WarehouseDockEnv(max_steps=32)
state_encoder = StateEncoder()
agent = QLearningAgent(state_encoder=state_encoder)

# Training loop
obs = env.reset()
done = False
while not done:
    action = agent.select_action(obs, training=True)
    next_obs, reward, done, info = env.step(action)
    agent.update(obs, action, reward, next_obs, done)
    obs = next_obs

# Evaluation
obs = env.reset()
done = False
while not done:
    action = agent.select_action(obs, training=False)
    obs, reward, done, info = env.step(action)
```

## Action Space

- **Action 0**: HOLD (skip assignment)
- **Action 1**: Assign front truck to Dock 0
- **Action 2**: Assign front truck to Dock 1
- **Action 3**: Assign front truck to Dock 2

Actions only succeed if:
1. The target dock is idle
2. There are waiting trucks in the queue

## Tests

Run all Q-Learning tests:

```bash
python -m unittest tests.test_qlearing_agent -v
```

**Coverage:**
- State encoder validation
- Q-table initialization
- Action selection (greedy + epsilon-greedy)
- Q-value updates
- Epsilon decay

## Learning Progression

| Training Stage | Epsilon | Avg Reward |
|---|---|---|
| Early (0-20 eps) | ~0.95 | 10.8 |
| Mid (40-60 eps) | ~0.78 | 15.0 |
| Late (80-100 eps) | ~0.64 | 18.8 |
| Evaluation (no explore) | 0.01 | 26.98 |

The agent shows steady improvement with declining exploration, indicating successful learning of the environment dynamics.

## Future Enhancements

1. **Deep Q-Learning (DQN)**: For larger state/action spaces
2. **Prioritized Experience Replay**: For faster convergence
3. **Double Q-Learning**: To reduce overestimation bias
4. **Policy Distillation**: To create compact inference models
5. **Multi-Agent Q-Learning**: For decentralized dock control

## Files

- `src/qlearing_agent.py`: Core agent implementation
- `tests/test_qlearing_agent.py`: Unit tests
- `scripts/train_ql_agent.py`: Full training pipeline
- `scripts/compare_agents.py`: Baseline comparison
