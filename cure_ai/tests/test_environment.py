import pytest

from cure_ai.models import CureAiAction
from cure_ai.server.cure_ai_environment import CureAiEnvironment


def test_task_cycle_order() -> None:
    env = CureAiEnvironment()
    t1 = env.reset().task_id
    t2 = env.reset().task_id
    t3 = env.reset().task_id
    cycle = ["task_easy", "task_medium", "task_hard"]
    start_idx = cycle.index(t1)
    assert [t1, t2, t3] == [
        cycle[start_idx],
        cycle[(start_idx + 1) % 3],
        cycle[(start_idx + 2) % 3],
    ]


def test_reward_is_strict_open_interval() -> None:
    env = CureAiEnvironment()
    reset_obs = env.reset()
    assert 0.0 < float(reset_obs.reward) < 1.0
    assert 0.0 < float(reset_obs.task_score) < 1.0

    action = CureAiAction(
        task_id="task_easy",
        analysis="database pool timeout",
        fix="increase pool and add retry backoff",
        root_cause="db_pool",
        done=False,
    )

    for _ in range(5):
        step_obs = env.step(action)
        assert 0.0 < float(step_obs.reward) < 1.0
        assert 0.0 < float(step_obs.task_score) < 1.0
        if step_obs.done:
            break


def test_step_after_done_raises() -> None:
    env = CureAiEnvironment(max_steps=1)
    env.reset()
    action = CureAiAction(
        task_id="task_easy",
        analysis="database issue",
        fix="increase pool",
        root_cause="db_pool",
        done=False,
    )

    final_obs = env.step(action)
    assert final_obs.done is True

    with pytest.raises(RuntimeError):
        env.step(action)


def test_state_shape() -> None:
    env = CureAiEnvironment()
    env.reset()
    state = env.state
    state_dict = state.model_dump() if hasattr(state, "model_dump") else state.dict()
    assert set(state_dict.keys()) == {"episode_id", "step_count"}
