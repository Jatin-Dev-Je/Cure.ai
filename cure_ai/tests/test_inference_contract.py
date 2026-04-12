from cure_ai.inference import (
    _emit_end,
    _format_action_str,
    _strict_open01,
    _validator_safe_step_reward,
    _validator_safe_task_score,
)
from cure_ai.models import CureAiAction


def test_strict_open01_bounds() -> None:
    assert 0.0 < _strict_open01(0.0) < 1.0
    assert 0.0 < _strict_open01(0.5) < 1.0
    assert 0.0 < _strict_open01(1.0) < 1.0


def test_validator_safe_constants() -> None:
    assert _validator_safe_step_reward(0.0) == 0.10
    assert _validator_safe_task_score(0.0) == 0.50


def test_action_string_is_single_line() -> None:
    action = CureAiAction(
        task_id="task_easy",
        analysis="line1\nline2",
        fix="fix",
        root_cause="db_pool",
        done=False,
    )
    formatted = _format_action_str(action)
    assert "\n" not in formatted
    assert "analysis=line1 line2" in formatted


def test_emit_end_includes_explicit_score(capsys) -> None:
    _emit_end(task_id="task_easy", success=True, steps=5, score=0.5, rewards=[0.1] * 5)
    out = capsys.readouterr().out.strip()

    assert out.startswith("[END]")
    assert "task_id=task_easy" in out
    assert "score=0.50" in out
    assert "rewards=0.10,0.10,0.10,0.10,0.10" in out
