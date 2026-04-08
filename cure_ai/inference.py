import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# Allow running as `python inference.py` from the environment directory.
try:
    from .client import CureAiEnv
    from .models import CureAiAction
except ImportError:  # pragma: no cover
    from cure_ai.client import CureAiEnv
    from cure_ai.models import CureAiAction


def _load_env_config() -> Dict[str, str]:
    agent_mode = os.environ.get("BASELINE_AGENT_MODE", "openai").strip().lower()
    api_base = os.environ.get("API_BASE_URL")
    model = os.environ.get("MODEL_NAME")
    hf_token = os.environ.get("HF_TOKEN")
    if agent_mode == "heuristic":
        return {
            "agent_mode": "heuristic",
            "api_base": api_base or "",
            "model": model or "heuristic-baseline",
            "hf_token": hf_token or "",
        }

    if not api_base or not model or not hf_token:
        missing: List[str] = []
        if not api_base:
            missing.append("API_BASE_URL")
        if not model:
            missing.append("MODEL_NAME")
        if not hf_token:
            missing.append("HF_TOKEN")
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return {
        "agent_mode": "openai",
        "api_base": api_base,
        "model": model,
        "hf_token": hf_token,
    }


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
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

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


def _heuristic_step(task_id: str, step: int) -> CureAiAction:
    if task_id == "task_easy":
        return CureAiAction(
            task_id=task_id,
            analysis="Database connection pool is exhausted causing timeout errors under peak traffic.",
            fix="Tune connection pool size, reduce connection churn, and add exponential retry backoff.",
            root_cause="db_pool",
            done=step >= 1,
        )
    if task_id == "task_medium":
        return CureAiAction(
            task_id=task_id,
            analysis="Cache is disabled and miss ratio spike is forcing traffic to origin, increasing latency.",
            fix="Re-enable cache feature flag, invalidate stale keys, and warm critical cache entries.",
            root_cause="cache_disabled",
            done=step >= 1,
        )
    return CureAiAction(
        task_id=task_id,
        analysis="Authentication is failing due to JWT signature and clock skew inconsistencies.",
        fix="Align JWT secret across services, sync clocks with NTP, and roll out auth config safely.",
        root_cause="jwt_secret_mismatch",
        done=step >= 1,
    )


def _emit_start(task_id: str, model: str, max_steps: int) -> None:
    print(
        "[START] "
        f"task_id={task_id} "
        f"model={model} "
        f"max_steps={max_steps}"
    )


def _emit_step(task_id: str, step: int, reward: float, done: bool, root_cause: str) -> None:
    print(
        "[STEP] "
        f"task_id={task_id} "
        f"step={step} "
        f"reward={reward:.4f} "
        f"done={str(done).lower()} "
        f"root_cause={root_cause}"
    )


def _emit_end(task_id: str, total_reward: float, steps: int, done: bool) -> None:
    print(
        "[END] "
        f"task_id={task_id} "
        f"total_reward={total_reward:.4f} "
        f"steps={steps} "
        f"done={str(done).lower()}"
    )


def main() -> None:
    config = _load_env_config()
    client: Optional[OpenAI] = None
    if config["agent_mode"] == "openai":
        client = _build_client(config["api_base"], config["hf_token"])

    results: List[Dict[str, float]] = []
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    env_base_url = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
    with CureAiEnv(base_url=env_base_url).sync() as env:
        expected_task_order = ["task_easy", "task_medium", "task_hard"]
        for expected_task_id in expected_task_order:
            reset_result = env.reset()
            task_id = reset_result.observation.task_id
            if task_id != expected_task_id:
                print(f"[WARN] expected_task={expected_task_id} observed_task={task_id}", file=sys.stderr)

            obs = reset_result.observation
            total_reward = 0.0
            _emit_start(task_id=task_id, model=config["model"], max_steps=obs.max_steps)

            for _step in range(obs.max_steps):
                if config["agent_mode"] == "heuristic":
                    action = _heuristic_step(task_id, obs.step)
                else:
                    if client is None:
                        raise RuntimeError("OpenAI client was not initialized for openai mode")
                    action = _llm_step(client, config["model"], task_id, obs)

                step_result = env.step(action)
                obs = step_result.observation
                reward = float(step_result.reward or 0.0)
                total_reward += reward

                _emit_step(
                    task_id=task_id,
                    step=obs.step,
                    reward=reward,
                    done=obs.done,
                    root_cause=(action.root_cause or "na"),
                )

                if obs.done:
                    break

            _emit_end(task_id=task_id, total_reward=total_reward, steps=obs.step, done=obs.done)
            results.append({"task_id": task_id, "total_reward": total_reward, "steps": obs.step})

    summary = {
        "run_id": run_id,
        "agent_mode": config["agent_mode"],
        "model": config["model"],
        "episodes": results,
        "mean_reward": sum(r["total_reward"] for r in results) / len(results),
    }

    results_path = Path(__file__).resolve().parent / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()