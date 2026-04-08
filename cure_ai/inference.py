import json
import os
import sys
from typing import Dict, List

from openai import OpenAI

# Allow running as `python inference.py` from the environment directory.
try:
    from .client import CureAiEnv
    from .models import CureAiAction
except ImportError:  # pragma: no cover
    from cure_ai.client import CureAiEnv
    from cure_ai.models import CureAiAction


def _load_env_config() -> Dict[str, str]:
    api_base = os.environ.get("API_BASE_URL")
    model = os.environ.get("MODEL_NAME")
    hf_token = os.environ.get("HF_TOKEN")
    if not api_base or not model or not hf_token:
        missing: List[str] = []
        if not api_base:
            missing.append("API_BASE_URL")
        if not model:
            missing.append("MODEL_NAME")
        if not hf_token:
            missing.append("HF_TOKEN")
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return {"api_base": api_base, "model": model, "hf_token": hf_token}


def _build_client(api_base: str, hf_token: str) -> OpenAI:
    return OpenAI(base_url=api_base, api_key=hf_token)


def _llm_step(
    client: OpenAI,
    model: str,
    task_id: str,
    observation,
) -> CureAiAction:
    system_prompt = (
        "You are an SRE assistant handling production incidents. "
        "Given an incident description, logs, and metrics, you must respond with:\n"
        "- analysis: concise reasoning about the root cause\n"
        "- fix: a safe, actionable remediation plan\n"
        "- root_cause: a very short label for the main cause\n"
        "Respond in JSON with keys: analysis, fix, root_cause, done (boolean)."
    )

    user_content = {
        "task_id": task_id,
        "description": observation.description,
        "logs": observation.logs,
        "metrics": observation.metrics,
        "step": observation.step,
        "max_steps": observation.max_steps,
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content)},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            "analysis": content,
            "fix": "",
            "root_cause": "",
            "done": False,
        }

    return CureAiAction(
        task_id=task_id,
        analysis=str(parsed.get("analysis", "")),
        fix=str(parsed.get("fix", "")),
        root_cause=str(parsed.get("root_cause", "")),
        done=bool(parsed.get("done", False)),
    )


def run_episode(
    env: CureAiEnv,
    client: OpenAI,
    model: str,
    task_id: str,
    max_steps: int = 5,
) -> Dict[str, float]:
    result = env.reset()
    obs = result.observation
    total_reward = 0.0

    for _ in range(max_steps):
        action = _llm_step(client, model, task_id, obs)
        step_result = env.step(action)
        obs = step_result.observation
        total_reward += step_result.reward or 0.0
        if obs.done:
            break

    return {
        "task_id": task_id,
        "total_reward": total_reward,
        "steps": obs.step,
    }


def main() -> None:
    config = _load_env_config()
    client = _build_client(config["api_base"], config["hf_token"])

    results: List[Dict[str, float]] = []
    env_base_url = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
    with CureAiEnv(base_url=env_base_url).sync() as env:
        # Evaluate each task once (environment cycles deterministically).
        for _ in range(3):
            reset_result = env.reset()
            task_id = reset_result.observation.task_id

            # Continue the episode from the already-reset state by reusing the observation.
            # (We step with a new client action loop to keep the run logic simple.)
            obs = reset_result.observation
            total_reward = 0.0
            for _step in range(obs.max_steps):
                action = _llm_step(client, config["model"], task_id, obs)
                step_result = env.step(action)
                obs = step_result.observation
                total_reward += step_result.reward or 0.0
                if obs.done:
                    break

            results.append({"task_id": task_id, "total_reward": total_reward, "steps": obs.step})

    summary = {
        "model": config["model"],
        "episodes": results,
        "mean_reward": sum(r["total_reward"] for r in results) / len(results),
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()