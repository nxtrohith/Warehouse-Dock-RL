---
title: {{title}}
emoji: {{emoji}}
colorFrom: {{colorFrom}}
colorTo: {{colorTo}}
sdk: {{sdk}}
sdk_version: "{{sdkVersion}}"
{{#pythonVersion}}
python_version: "{{pythonVersion}}"
{{/pythonVersion}}
app_file: app.py
pinned: false
---

# Warehouse Dock Scheduling Environment

## 1. Project Title

Warehouse Dock Scheduling Environment

## 2. Overview

Warehouse operations often suffer from dock bottlenecks, truck queues, and inefficient dispatch timing. This project provides an OpenEnv-compatible reinforcement learning environment where an AI agent learns to schedule trucks to available docks.

The environment simulates realistic dock utilization decisions with finite time, queue pressure, and action constraints. The objective is to reduce waiting time, increase throughput, and maintain stable dock usage under varying load conditions.

## 3. Features

- OpenEnv-compatible environment interface
- Standard API methods: step(), reset(), state()
- Real-world warehouse scheduling simulation
- Reward-based learning signals at each step
- Multiple difficulty tasks: easy, medium, hard
- Deterministic graders for reproducible evaluation

## 4. Environment Design

### Observation Space

At each step, the agent observes:

- waiting_trucks: number of trucks waiting in queue
- dock_status: per-dock availability (idle or busy)
- unloading_times: remaining unload time at each dock
- time_remaining: steps left in the episode

Example observation:

~~~json
{
  "waiting_trucks": 4,
  "queue_unload_times": [5, 3, 7, 0, 0],
  "dock_status": [1, 0, 1],
  "unloading_times": [2, 0, 4],
  "time_remaining": 21
}
~~~

### Action Space

The action space is discrete:

- 0: hold (do nothing)
- 1..N: assign front truck in queue to a specific dock

In the default setup:

- 0: hold
- 1: assign to dock 0
- 2: assign to dock 1
- 3: assign to dock 2

### Reward Function

The reward is shaped throughout the episode, not only at termination.

- Positive rewards:
- Valid truck-to-dock assignments
- Completed unload operations

- Penalties:
- Invalid actions
- Idle docks while trucks are waiting
- Queue delay / accumulated waiting pressure
- Holding when a valid assignment is available

This dense reward design encourages fast, valid, and throughput-oriented scheduling.

### Episode Termination

An episode ends when either condition is met:

- Maximum step limit reached
- All trucks have been processed (queue empty and docks idle)

## 5. Tasks

The benchmark includes three deterministic tasks:

- Easy: small queue and few decisions; learn valid assignment behavior quickly
- Medium: moderate load; balance queue reduction with dock utilization
- Hard: higher scheduling complexity with tighter efficiency requirements and stochastic pressure

What the agent must achieve:

- Avoid invalid assignments
- Keep docks utilized when trucks are waiting
- Minimize waiting queue over time
- Maximize completion throughput within episode limits

## 6. Grading System

- Deterministic scoring for repeatable evaluation
- Score range: 0.0 to 1.0
- Score reflects efficiency and correctness
- Uses task-specific grader functions with clear acceptance criteria

## 7. Installation and Setup

~~~bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
~~~

Optional:

- Create and activate a virtual environment before installation

## 8. Running the Environment

Run a quick random-action smoke test:

~~~bash
python scripts/smoke_run.py
~~~

Run test suite:

~~~bash
pytest -q
~~~

Run the inference script:

~~~bash
python inference.py
~~~

Run the HTTP API server (OpenEnv-style reset/step interface):

~~~bash
python -m server.app
~~~

## 9. Docker Setup

~~~bash
docker build -t warehouse-openenv .
docker run warehouse-openenv
~~~

## 10. Inference Script

The inference pipeline supports an OpenAI-compatible API endpoint for action selection.

Required environment variables:

- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Common optional variables include TASK_NAME, MAX_STEPS, ENV_SEED, and POLICY_MODE.

## 11. Example Output

~~~text
[START]
[STEP]
[STEP]
[END]
~~~

## 12. Project Structure

~~~text
env/
inference.py
Dockerfile
openenv.yaml
README.md
~~~

## 13. Future Improvements

- Multi-agent dock coordination support
- Dynamic truck arrival processes with richer traffic patterns
- Priority-aware scheduling for urgent or SLA-bound loads
