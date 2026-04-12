# Cure AI: Round-1 Rubric Mapping

This document maps the current implementation to judging criteria and provides evidence pointers.

## 1) Real-World Utility (30%)

### Problem
Production incident triage is noisy, time-sensitive, and safety-critical. Existing benchmarks underrepresent realistic SRE workflows with logs, metrics, and constrained remediation.

### What Cure AI adds
- Three incident classes that mirror real on-call patterns:
  - DB connection pool exhaustion
  - Cache misconfiguration latency cascade
  - JWT/auth misconfiguration outage
- Typed action schema requiring analysis, fix plan, and root cause label.
- Safety-aware evaluation with explicit penalties for destructive remediation patterns.

### Evidence
- Task definitions and telemetry context: cure_ai/server/cure_ai_environment.py
- Typed interfaces: cure_ai/models.py
- End-to-end baseline execution: cure_ai/inference.py

## 2) Task & Grader Quality (25%)

### Coverage and difficulty
- 3 tasks with increasing operational complexity and ambiguity.
- Hard task includes multiple overlapping signals (JWT signature + clock skew + user-facing symptoms).

### Determinism and reproducibility
- Deterministic task cycling across episodes.
- Deterministic grader logic based on explicit keyword/safety checks.

### Score validity
- All task score-bearing fields are constrained to strict open interval (0,1).
- Inference emits explicit per-task END score for parser clarity.

### Evidence
- Grader logic and penalties: cure_ai/server/cure_ai_environment.py
- Strict score constraints: cure_ai/models.py
- Parser-safe END score emission: cure_ai/inference.py

## 3) Environment Design (20%)

### API quality
- OpenEnv-compatible reset/step/state flow.
- Rich observations: task description, logs, metrics, feedback message, reward breakdown.

### Episode boundaries
- Explicit done handling and step-after-done guard.

### Evidence
- Environment server implementation: cure_ai/server/cure_ai_environment.py
- API wiring: cure_ai/server/app.py

## 4) Code Quality & Spec Compliance (15%)

### Compliance status
- openenv validate path documented and operational.
- Dockerized deployment path documented.
- HF Space endpoint integration validated.

### Reliability practices
- Defensive parsing in inference output extraction.
- Runtime-safe fallbacks for transient LLM errors.

### Evidence
- Validation script: cure_ai/validate-submission.sh
- Docker setup: cure_ai/server/Dockerfile
- Baseline runner: cure_ai/inference.py

## 5) Creativity & Novelty (10%)

### Novelty thesis
- Incident-response RL with safety-aware remediation quality is underrepresented in standard OpenEnv examples.
- Benchmark focuses on practical operator behavior: diagnosis quality, fix safety, and root-cause precision.

## Suggested Judge Talking Points
- "This environment benchmarks incident reasoning quality, not only answer correctness."
- "It rewards safe remediation and penalizes operationally dangerous suggestions."
- "It provides realistic telemetry context and reproducible task progression for fair agent evaluation."
