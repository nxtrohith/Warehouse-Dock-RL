"""Inference runner with strict stdout logging for OpenEnv-style evaluation."""

from __future__ import annotations

import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Mapping, Optional

from openai import OpenAI

from src.config import ACTION_HOLD, VALID_ACTIONS
from src.env import WarehouseDockEnv
from src.task_graders import task_1_grader, task_2_grader, task_3_grader


def _load_local_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file if present."""
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if not key or key in os.environ:
                    continue

                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]

                os.environ[key] = value
    except OSError:
        # Best-effort loading only; explicit env vars still work.
        return


_load_local_dotenv()


# Required by organizer environment configuration.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# Only these two values have defaults.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mixtral-8x7b-32768")

TASK_NAME = os.getenv("TASK_NAME", "task_1")
BENCHMARK = os.getenv("BENCHMARK", "warehouse_dock")
MAX_STEPS = int(os.getenv("MAX_STEPS", "32"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "64"))
VALID_TASK_NAMES = {"task_1", "task_2", "task_3"}


SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a warehouse dock scheduler with discrete actions.
    Return exactly one action integer: 0, 1, 2, or 3.
    Action 0 means HOLD.
    Actions 1..3 assign the front truck to dock index action-1.
    Reply with only the integer and no extra text.
    """
).strip()


def _as_obs_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return dict(obs.model_dump())
    return dict(obs)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(step: int, obs: Mapping[str, Any], history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation:
        - waiting_trucks: {obs.get("waiting_trucks")}
        - queue_unload_times: {obs.get("queue_unload_times")}
        - dock_status: {obs.get("dock_status")}
        - unloading_times: {obs.get("unloading_times")}
        - time_remaining: {obs.get("time_remaining")}

        Recent history:
        {history_block}

        Select the single best next action integer.
        """
    ).strip()


def request_action_text(client: OpenAI, step: int, obs: Mapping[str, Any], history: List[str]) -> str:
    prompt = build_user_prompt(step, obs, history)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return (completion.choices[0].message.content or "").strip()


def parse_action(action_text: str) -> tuple[int, Optional[str]]:
    text = action_text.strip()
    match = re.search(r"-?\d+", text)
    if not match:
        return ACTION_HOLD, "unparseable_action_output"

    raw_action = int(match.group(0))
    if raw_action not in VALID_ACTIONS:
        return ACTION_HOLD, f"invalid_action_output:{raw_action}"
    return raw_action, None


def _score_for_task(task_name: str, metrics: Mapping[str, Any]) -> float:
    if task_name == "task_1":
        return task_1_grader(metrics)
    if task_name == "task_2":
        return task_2_grader(metrics)
    if task_name == "task_3":
        return task_3_grader(metrics)
    raise ValueError(f"Unsupported TASK_NAME '{task_name}'. Expected one of: {sorted(VALID_TASK_NAMES)}")


def run_episode() -> None:
    client: Optional[OpenAI] = None
    env: Optional[WarehouseDockEnv] = None
    exit_code = 0

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    invalid_action_count = 0
    assigned_within_first_n_steps = 0
    waiting_trucks_total = 0.0
    idle_dock_steps = 0
    baseline_reward = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not API_KEY:
            raise RuntimeError("Missing API key: set API_KEY, GROQ_API_KEY, HF_TOKEN, or OPENAI_API_KEY")
        if TASK_NAME not in VALID_TASK_NAMES:
            raise RuntimeError(f"Invalid TASK_NAME '{TASK_NAME}'. Expected one of: {sorted(VALID_TASK_NAMES)}")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = WarehouseDockEnv(max_steps=MAX_STEPS)

        obs = env.reset()
        obs_dict = _as_obs_dict(obs)
        initial_queue_size = int(obs_dict.get("waiting_trucks", 0))
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            last_action_error: Optional[str] = None
            action_text = "0"

            try:
                action_text = request_action_text(client, step, obs_dict, history)
            except Exception as exc:  # Keep runtime alive and continue with HOLD.
                action_text = "0"
                last_action_error = " ".join(str(exc).split())

            action, parse_error = parse_action(action_text)
            if parse_error:
                last_action_error = parse_error

            next_obs, reward, done, info = env.step(action)
            next_obs_dict = _as_obs_dict(next_obs)

            if action != ACTION_HOLD and not bool(info.get("assigned", False)):
                invalid_action_count += 1
                if last_action_error is None:
                    last_action_error = "invalid_assignment"

            if step <= 2 and bool(info.get("assigned", False)):
                assigned_within_first_n_steps = 1

            waiting_now = float(next_obs_dict.get("waiting_trucks", 0.0))
            waiting_trucks_total += waiting_now
            if waiting_now > 0:
                idle_dock_steps += sum(1 for x in next_obs_dict.get("dock_status", []) if int(x) == 0)

            rewards.append(float(reward))
            steps_taken = step

            log_step(
                step=step,
                action=str(action),
                reward=float(reward),
                done=bool(done),
                error=last_action_error,
            )

            history.append(f"step={step} action={action} reward={float(reward):.2f}")
            obs_dict = next_obs_dict

        final_state = env.state()
        completed_trucks = int(final_state.processed_trucks)
        total_trucks_created = int(final_state.total_trucks_created)
        queue_remaining = int(final_state.waiting_trucks)

        mean_waiting = waiting_trucks_total / max(1, steps_taken)
        average_reward = sum(rewards) / max(1, steps_taken)

        metrics: Dict[str, Any] = {
            "invalid_action_count": invalid_action_count,
            "assigned_within_first_n_steps": assigned_within_first_n_steps,
            "completed_trucks": completed_trucks,
            "initial_queue_size": initial_queue_size,
            "mean_waiting_trucks": mean_waiting,
            "idle_dock_steps": idle_dock_steps,
            "max_waiting_trucks": max(initial_queue_size, 1),
            "average_reward": average_reward,
            "baseline_reward": baseline_reward,
            "total_trucks_created": max(total_trucks_created, 1),
            "queue_remaining": queue_remaining,
        }

        score = _score_for_task(TASK_NAME, metrics)
        score = max(0.0, min(1.0, float(score)))
        success = score >= 0.5

    except Exception as exc:
        exit_code = 1
        err = " ".join(str(exc).split())
        print(f"[ERROR] {err}", file=sys.stderr, flush=True)

    finally:
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as close_exc:
                    print(f"[WARN] close_failed={' '.join(str(close_exc).split())}", file=sys.stderr, flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    run_episode()