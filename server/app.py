"""HTTP server for warehouse dock environment with /reset and /step endpoints."""

from __future__ import annotations

import os
from threading import Lock
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.env import WarehouseDockEnv

app = FastAPI(title="Warehouse Dock OpenEnv API")

_env_lock = Lock()
_env: Optional[WarehouseDockEnv] = None


class ResetRequest(BaseModel):
    seed: int = Field(default=7, description="Random seed for deterministic episodes.")
    max_steps: int = Field(default=32, ge=1, le=10000, description="Episode horizon.")


class StepRequest(BaseModel):
    action: int = Field(..., description="Action integer in [0, 3].")


def _as_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return dict(value)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Use POST /reset and POST /step."}


@app.post("/reset")
def reset(payload: ResetRequest) -> Dict[str, Any]:
    global _env
    with _env_lock:
        _env = WarehouseDockEnv(seed=payload.seed, max_steps=payload.max_steps)
        observation = _env.reset()
        return {
            "observation": _as_dict(observation),
            "state": _as_dict(_env.state()),
            "done": False,
        }


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    with _env_lock:
        if _env is None:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /reset first.")

        observation, reward, done, info = _env.step(payload.action)
        return {
            "observation": _as_dict(observation),
            "reward": float(reward),
            "done": bool(done),
            "info": dict(info),
            "state": _as_dict(_env.state()),
        }


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
