# Judging Checklist (Safe to Use Before Resubmission)

This checklist is designed to improve presentation quality without changing runtime behavior.

## A. Validation Safety
- [ ] Do not modify runtime scoring contract unless strictly necessary.
- [ ] Confirm `openenv validate` passes.
- [ ] Confirm HF Space `/reset` responds 200.
- [ ] Confirm inference emits START/STEP/END lines.

## B. Score Contract Safety
- [ ] Ensure task scores are strictly in (0,1).
- [ ] Ensure END line includes explicit `task_id` and `score`.
- [ ] Ensure no post-done scoring path returns out-of-range values.

## C. Judge-Facing Story
- [ ] Problem statement is clear and real-world.
- [ ] Three-task difficulty ladder is explained.
- [ ] Safety penalties and deterministic grading are explained.
- [ ] Demo flow is practiced (5-7 min).

## D. Evidence Pack
- [ ] Keep links ready:
  - GitHub repo URL
  - HF Space URL
  - Rubric mapping doc
  - Demo script
- [ ] Keep one-paragraph novelty claim ready.

## E. Final Submission Hygiene
- [ ] Confirm branch clean before submit.
- [ ] Avoid last-minute runtime edits unless absolutely required.
- [ ] If editing docs only, mention "no runtime behavior changes" in commit message.
