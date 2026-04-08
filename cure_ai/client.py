# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cure Ai Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CureAiAction, CureAiObservation


class CureAiEnv(
    EnvClient[CureAiAction, CureAiObservation, State]
):
    """
    Client for the Cure Ai Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CureAiEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.description)
        ...
        ...     result = client.step(
        ...         CureAiAction(
        ...             analysis="Likely DB pool exhaustion due to high concurrency.",
        ...             fix="Increase pool limits safely and add exponential backoff.",
        ...             root_cause="db_pool",
        ...             done=True,
        ...         )
        ...     )
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CureAiEnv.from_docker_image("cure_ai-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CureAiAction(analysis="Inspect logs", fix="Collect metrics"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CureAiAction) -> Dict:
        """
        Convert CureAiAction to JSON payload for step message.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[CureAiObservation]:
        """
        Parse server response into StepResult[CureAiObservation].
        """
        obs_data = payload.get("observation", {})
        observation = CureAiObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
