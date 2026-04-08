"""
Cure.ai — OpenEnv RL Environment

Typed models for Action and Observation.
Designed for production-grade evaluation and OpenEnv compliance.
"""

from typing import List, Dict
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CureAiAction(Action):
    """
    Action sent by the agent.

    The agent analyzes the incident and provides:
    - analysis: reasoning
    - fix: solution
    - root_cause: concise label
    """

    task_id: str = Field(
        default="task_easy",
        description="Task identifier: task_easy | task_medium | task_hard"
    )

    analysis: str = Field(
        default="",
        description="Agent's reasoning about the issue"
    )

    fix: str = Field(
        default="",
        description="Actionable fix proposed by the agent"
    )

    root_cause: str = Field(
        default="",
        description="Short root cause label"
    )

    done: bool = Field(
        default=False,
        description="Indicates if agent has completed response"
    )


class CureAiObservation(Observation):
    """
    Observation returned to the agent.

    Contains:
    - problem description
    - logs + metrics
    - reward signal
    """

    task_id: str = Field(
        default="task_easy",
        description="Current task id"
    )

    description: str = Field(
        default="",
        description="Incident description"
    )

    logs: List[str] = Field(
        default_factory=list,
        description="System logs"
    )

    metrics: Dict = Field(
        default_factory=dict,
        description="System metrics like latency, error rate"
    )

    step: int = Field(
        default=0,
        description="Current step number"
    )

    max_steps: int = Field(
        default=5,
        description="Maximum allowed steps"
    )

    reward: float = Field(
        default=0.0,
        description="Reward signal for this step (0.0 - 1.0)"
    )

    done: bool = Field(
        default=False,
        description="Whether the episode has terminated"
    )

    message: str = Field(
        default="",
        description="Feedback message from environment / grader"
    )