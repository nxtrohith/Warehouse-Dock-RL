"""Configuration for the warehouse dock scheduling RL environment."""

MAX_STEPS = 32
NUM_DOCKS = 3

# Observation settings
QUEUE_FEATURE_SIZE = 5
INITIAL_QUEUE_SIZE = 6

# Processing times (in simulation ticks)
MIN_UNLOAD_TICKS = 2
MAX_UNLOAD_TICKS = 6

# Action space
ACTION_HOLD = 0
# Actions 1..NUM_DOCKS map to assigning the front truck to dock index (action - 1).
VALID_ACTIONS = tuple(range(NUM_DOCKS + 1))

# Reward weights
REWARD_VALID_ASSIGN = 1.0
REWARD_COMPLETION = 5.0
PENALTY_WAIT_PER_TRUCK = -0.1
PENALTY_IDLE_DOCK_WHILE_QUEUE = -0.5
PENALTY_INVALID_ACTION = -2.0
REWARD_HOLD_NO_VALID_ASSIGN = 0.2
PENALTY_HOLD_WHEN_ASSIGN_AVAILABLE = -0.5

# Episode behavior
DONE_WHEN_ALL_PROCESSED = True

# Optional arrivals for non-stationary load.
ENABLE_ARRIVALS = False
ARRIVAL_PROBABILITY = 0.35
