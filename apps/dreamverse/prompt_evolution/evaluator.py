"""
OpenEvolve evaluator for the Dreamverse `rewrite_new_rollout` system prompt.

What this evaluates
-------------------
The candidate program file is a SYSTEM PROMPT (markdown) for the production
"rewriter" LLM. For each user idea in eval_bank.jsonl we:

  1. Strip the EVOLVE-BLOCK marker lines, yielding the real system prompt.
  2. Run the *actual* dreamverse rewrite path (PromptEnhancer.rewrite_prompt_sequence,
     new-rollout mode) with that system prompt -> a 6-segment rollout from the
     production task model (gpt-oss-120b on Cerebras/Groq).
  3. Structurally validate the rollout (JSON ok, exactly 6 non-empty segments).
  4. Score it with a frontier LLM-as-judge on a seam-centered rubric.

This is the cheap Stage-1 proxy: NO video render. It exists so the evolution
loop iterates in seconds-to-minutes. Stage 2 (render + boundary slicing +
human/video eval via EvolveService) is a documented TODO at the bottom.

Two/three LLMs (do not confuse):
  - EVOLVER LLM: mutates the prompt; configured in config.yaml `llm:` (OpenEvolve uses it).
  - TASK LLM:    gpt-oss-120b @ Cerebras/Groq; configured via FASTVIDEO_PROMPT_* env
                 + the dreamverse PromptEnhancer. This is what we optimize the prompt FOR.
  - JUDGE LLM:   frontier model scoring rollouts; JUDGE_* env, defaults to the evolver llm.

Required environment
--------------------
  OPENAI_API_KEY              key for the frontier judge (and evolver) endpoint
  CEREBRAS_API_KEY / GROQ_API_KEY   key for the production task model
  (optional) DREAMVERSE_ROOT  path to apps/dreamverse (default: known workspace path)
  (optional) JUDGE_MODEL / JUDGE_API_BASE / JUDGE_API_KEY   override judge LLM
  (optional) EVAL_BANK_LIMIT  cap number of eval cases (default: all)
  (optional) EVAL_K           samples per case to tame temperature variance (default: 1)
"""

import asyncio
import json
import os
import sys
import traceback
from typing import Any

import yaml
from openai import OpenAI

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover - allows standalone testing
    EvaluationResult = None

HERE = os.path.dirname(os.path.abspath(__file__))

# --- config / env -----------------------------------------------------------
with open(os.path.join(HERE, "config.yaml")) as f:
    _CFG = yaml.safe_load(f)
_LLM = _CFG.get("llm", {})

# This file lives at apps/dreamverse/prompt_evolution/, so the `dreamverse`
# package is one level up. Override with DREAMVERSE_ROOT if run from elsewhere.
DREAMVERSE_ROOT = os.environ.get("DREAMVERSE_ROOT", os.path.dirname(HERE))
EVAL_BANK_PATH = os.path.join(HERE, "eval_bank.jsonl")
EVAL_K = int(os.environ.get("EVAL_K", "1"))
_limit_env = os.environ.get("EVAL_BANK_LIMIT")
EVAL_BANK_LIMIT = int(_limit_env) if _limit_env else None

# Judge LLM: default to reusing the evolver (frontier) model from config.yaml.
_models = _LLM.get("models", [])
_default_model = _models[0]["name"] if _models else _LLM.get("primary_model", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", _default_model)
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", _LLM.get("api_base", "https://api.openai.com/v1"))
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Rubric weights for the Stage-1 combined score.
RUBRIC_WEIGHTS = {
    "seam_continuity": 0.40,
    "instruction_follow": 0.30,
    "variety": 0.20,
    "house_style": 0.10,
}
EXPECTED_SEGMENTS = 6


# --- prompt loading ---------------------------------------------------------
def load_system_prompt(program_path: str) -> str:
    """Read the candidate file and strip EVOLVE-BLOCK markers + the guidance
    HTML comment so the marker lines never leak into the real system prompt."""
    with open(program_path) as f:
        lines = f.readlines()
    out, in_guidance = [], False
    for line in lines:
        s = line.strip()
        if "EVOLVE-BLOCK-START" in s or "EVOLVE-BLOCK-END" in s:
            continue
        # drop the multi-line guidance HTML comment that explains the evolve region
        if s == "<!--":
            in_guidance = True
            continue
        if in_guidance:
            if s == "-->":
                in_guidance = False
            continue
        out.append(line)
    return "".join(out).strip()


# --- dreamverse task model --------------------------------------------------
_ENHANCER = None


def _get_enhancer() -> Any:
    global _ENHANCER
    if _ENHANCER is None:
        if DREAMVERSE_ROOT not in sys.path:
            sys.path.insert(0, DREAMVERSE_ROOT)
        from dreamverse.prompt_enhancer import PromptEnhancer  # type: ignore

        _ENHANCER = PromptEnhancer()
    return _ENHANCER


async def _run_rollout(system_prompt: str, user_idea: str) -> Any:
    enhancer = _get_enhancer()
    return await enhancer.rewrite_prompt_sequence(
        [],  # new-rollout: no existing prompts
        rewrite_instruction=user_idea,
        system_prompt_override=system_prompt,
    )


def run_rollout(system_prompt: str, user_idea: str) -> Any:
    return asyncio.run(_run_rollout(system_prompt, user_idea))


def validate_rollout(result: Any) -> tuple[bool, str]:
    if result is None:
        return False, "no result"
    if getattr(result, "error", None):
        return False, f"task error: {result.error}"
    if getattr(result, "fallback_used", False):
        return False, "fallback path used (task model did not produce a valid rollout)"
    prompts = list(getattr(result, "prompts", []) or [])
    if len(prompts) != EXPECTED_SEGMENTS:
        return False, f"expected {EXPECTED_SEGMENTS} segments, got {len(prompts)}"
    if any(not (p or "").strip() for p in prompts):
        return False, "one or more empty segments"
    return True, "ok"


# --- frontier judge ---------------------------------------------------------
_JUDGE = None

JUDGE_SYSTEM = """You are a strict evaluator of AI-generated video "rollout" plans.
A rollout is 6 sequential 5-second segment prompts (30s total) for a video model
where each segment is conditioned ONLY on the last frames/audio of the previous
segment. Judge the WRITTEN plan only (no video exists).

Score these axes in [0,1] (1 = excellent):
- seam_continuity: do adjacent segments hand off plausibly? subjects/scene/lighting
  carry forward; no off-screen subjects reappearing without justification; stable end frames.
- instruction_follow: does the rollout faithfully cover the user's idea, tone, and intent?
- variety: is there real progression and motion across segments, NOT static repetition
  or wallpaper. Low score if segments are near-duplicates or nothing happens.
- house_style: readable single beat per segment, dialogue used where natural, concrete
  visible/audible detail, no internal-thought narration.

Also set too_static=true if the rollout is essentially motionless / repetitive.
Return ONLY JSON: {"seam_continuity":0-1,"instruction_follow":0-1,"variety":0-1,
"house_style":0-1,"too_static":true|false,"note":"one short sentence"}"""


def _get_judge() -> OpenAI:
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = OpenAI(base_url=JUDGE_API_BASE, api_key=JUDGE_API_KEY)
    return _JUDGE


def judge_rollout(user_idea: str, segments: list[str]) -> dict:
    client = _get_judge()
    payload = {"user_idea": user_idea, "segments": segments}
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False)
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    out: dict[str, Any] = {}
    for k in RUBRIC_WEIGHTS:
        try:
            out[k] = max(0.0, min(1.0, float(data.get(k, 0.0))))
        except (TypeError, ValueError):
            out[k] = 0.0
    out["too_static"] = bool(data.get("too_static", False))
    out["note"] = str(data.get("note", ""))[:200]
    return out


def _rubric_score(r: dict[str, Any]) -> float:
    return sum(w * float(r.get(k, 0.0)) for k, w in RUBRIC_WEIGHTS.items())


# --- main stage-1 evaluation -------------------------------------------------
def _load_bank() -> list[dict]:
    cases = []
    with open(EVAL_BANK_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:EVAL_BANK_LIMIT] if EVAL_BANK_LIMIT else cases


def evaluate_stage1(program_path: str) -> Any:
    system_prompt = load_system_prompt(program_path)
    prompt_length = float(len(system_prompt))  # raw continuous (latency/cost proxy)

    cases = _load_bank()
    scores: list[float] = []
    static_flags: list[float] = []
    valid_flags: list[float] = []
    latencies: list[float] = []
    rubric_acc: dict[str, list[float]] = {k: [] for k in RUBRIC_WEIGHTS}
    malformed_examples: list[str] = []
    weak_seam_examples: list[str] = []

    for case in cases:
        for _ in range(EVAL_K):
            try:
                result = run_rollout(system_prompt, case["prompt"])
            except Exception as e:  # task model / import failure
                valid_flags.append(0.0)
                scores.append(0.0)
                static_flags.append(1.0)
                malformed_examples.append(f"[{case['id']}] exception: {e}")
                continue

            latencies.append(float(getattr(result, "latency_ms", 0.0) or 0.0))
            ok, reason = validate_rollout(result)
            if not ok:
                valid_flags.append(0.0)
                scores.append(0.0)  # hard guard: malformed output is worthless
                static_flags.append(1.0)
                raw = (getattr(result, "raw_response_text", "") or "")[:300]
                malformed_examples.append(f"[{case['id']}] {reason} :: {raw}")
                continue

            valid_flags.append(1.0)
            try:
                r = judge_rollout(case["prompt"], list(result.prompts))
            except Exception as e:
                # judge failed: keep it valid but neutral, log it
                scores.append(0.5)
                static_flags.append(0.0)
                malformed_examples.append(f"[{case['id']}] judge error: {e}")
                continue

            s = _rubric_score(r)
            scores.append(s)
            static_flags.append(1.0 if r["too_static"] else 0.0)
            for k in RUBRIC_WEIGHTS:
                rubric_acc[k].append(r[k])
            if r["seam_continuity"] < 0.5 or r["too_static"]:
                weak_seam_examples.append(f"[{case['id']}] seam={r['seam_continuity']:.2f} "
                                          f"static={r['too_static']} :: {r['note']}")

    def _mean(xs: list[float], default: float = 0.0) -> float:
        return sum(xs) / len(xs) if xs else default

    metrics = {
        "combined_score": _mean(scores),
        "prompt_length": prompt_length,
        "static_rate": _mean(static_flags),
        "valid_rate": _mean(valid_flags),
        "seam_continuity": _mean(rubric_acc["seam_continuity"]),
        "instruction_follow": _mean(rubric_acc["instruction_follow"]),
        "variety": _mean(rubric_acc["variety"]),
        "house_style": _mean(rubric_acc["house_style"]),
        "avg_task_latency_ms": _mean(latencies),
    }

    if EvaluationResult is None:
        return metrics

    artifacts = {}
    if malformed_examples:
        artifacts["malformed_examples"] = "\n".join(malformed_examples[:8])
    if weak_seam_examples:
        artifacts["weak_seam_examples"] = "\n".join(weak_seam_examples[:8])
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate(program_path: str) -> Any:
    """Backward-compatible entrypoint (cascade disabled -> this is called)."""
    try:
        return evaluate_stage1(program_path)
    except Exception as e:
        metrics = {"combined_score": 0.0, "prompt_length": 0.0, "static_rate": 1.0, "valid_rate": 0.0}
        if EvaluationResult is None:
            return metrics
        return EvaluationResult(
            metrics=metrics,
            artifacts={
                "stderr": str(e),
                "traceback": traceback.format_exc()[:2000]
            },
        )


# =============================================================================
# TODO Stage 2 (sparse, expensive) -- wire when the eval side is ready.
#
#   def evaluate_stage2(program_path):
#       """Render rollouts + slice 5 boundary windows + human/video eval.
#       Bridge to apps/dreamverse's port of EvolveService:
#         svc.create_candidate(prompt_surface="rewrite_new_rollout",
#                              system_prompt_text=load_system_prompt(program_path), ...)
#         svc.create_rollout(...) ; svc.slice_rollout_boundaries(...)
#         ... collect human pairwise labels / WER+video metrics ...
#         snap = svc.recompute_fitness(candidate_id)
#         return EvaluationResult(metrics={"combined_score": snap["final_fitness"], ...})
#       """
#
# Then in config.yaml set: cascade_evaluation: true ; cascade_thresholds: [0.6]
# so only Stage-1 survivors reach the expensive stage.
# =============================================================================

if __name__ == "__main__":
    # Standalone smoke test: python evaluator.py [path]
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "rewrite_new_rollout_system_prompt.md")
    res = evaluate(path)
    print(json.dumps(getattr(res, "metrics", res), indent=2))
