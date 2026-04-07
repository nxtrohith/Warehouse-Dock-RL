"""Typed models for OpenEnv compliance using Pydantic."""

from pydantic import BaseModel, Field
from typing import List


class Observation(BaseModel):
    waiting_trucks: int = Field(..., description="Number of trucks waiting in the queue.")
    queue_unload_times: List[int] = Field(..., description="Unload times for the next K trucks in the queue.")
    dock_status: List[int] = Field(..., description="Status of each dock (0 = idle, 1 = busy).")
    unloading_times: List[int] = Field(..., description="Remaining unload times for trucks at each dock.")
    time_remaining: int = Field(..., description="Number of steps remaining in the episode.")


class State(Observation):
    current_step: int = Field(..., description="Current step in the episode.")
    processed_trucks: int = Field(..., description="Number of trucks processed so far.")
    total_trucks_created: int = Field(..., description="Total number of trucks created in the episode.")


class StepResponse(BaseModel):
    observation: Observation
    reward: float = Field(..., description="Reward received after taking the action.")
    done: bool = Field(..., description="Whether the episode has ended.")
    info: dict = Field(..., description="Additional debugging information.")