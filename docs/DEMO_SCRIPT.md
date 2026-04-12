# Cure AI Demo Script (5-7 Minutes)

## Goal
Demonstrate practical value: safer and faster incident triage with reproducible evaluation.

## 0:00-0:45 — Problem Framing
- "On-call teams lose time in noisy incidents: logs are ambiguous, fixes can be unsafe, and evaluation is inconsistent."
- "Cure AI is an OpenEnv benchmark that evaluates diagnosis quality, remediation safety, and root-cause accuracy."

## 0:45-1:45 — Environment Walkthrough
- Show the three tasks and their escalation:
  - Easy: DB pool exhaustion
  - Medium: cache misconfiguration cascade
  - Hard: auth/JWT outage with mixed signals
- Emphasize observation payload:
  - description
  - logs
  - metrics

## 1:45-3:30 — Live Interaction
- Start environment and run baseline inference.
- Highlight structured evaluator logs:
  - START line for task context
  - STEP lines with action + reward + done
  - END line with explicit task_id and score
- Call out safety behavior:
  - destructive fix patterns are penalized

## 3:30-4:45 — Evaluation Design
- Explain deterministic grader:
  - analysis signal
  - remediation signal
  - root-cause signal
  - safety penalty
- Explain strict score contract and parser-safe output for reproducible benchmarking.

## 4:45-6:00 — Why It Matters
- Real-world utility:
  - incident-response quality benchmark
  - safety-aware scoring
  - reproducible evaluation protocol
- Value for community:
  - enables better comparison of agentic incident-response policies.

## 6:00-7:00 — Close
- "Cure AI bridges practical SRE incidents and RL-style evaluation with safety-first grading."
- "It is production-inspired, deterministic, and ready for comparative agent benchmarking."

## Demo Commands

```bash
cd cure_ai
uv run server --host 0.0.0.0 --port 8000
```

```bash
cd cure_ai
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=your_hf_token \
ENV_BASE_URL=http://localhost:8000 \
python inference.py
```
