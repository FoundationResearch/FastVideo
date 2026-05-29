# Dreamverse `rewrite_new_rollout` prompt evolution

Evolve the **system prompt** of the Dreamverse "rewriter" LLM — the one that
turns a one-line user idea into a JSON rollout of **6 sequential 5-second
segment prompts** (30s) for the ltx2 video model. Target surface: the
`apps/dreamverse` `rewrite_user` path (`rewrite_prompt_sequence`, new-rollout mode).

This is **Phase 2 ("Connect OpenEvolve")** of Will's `evolve_plan.md`. The
evaluation/annotation/fitness infrastructure (Phases 0/1) is the eval side
(`will/oe` `EvolveService`) and is owned separately; here we only build the
OpenEvolve harness that mutates the prompt and consumes a fitness signal.

## The three LLMs (do not confuse)

| Role | What | Configured in |
|------|------|---------------|
| **Evolver** | mutates the system prompt | `config.yaml` `llm:` (frontier model) |
| **Task** | gpt-oss-120b @ Cerebras/Groq, runs the candidate prompt | `FASTVIDEO_PROMPT_*` env + dreamverse `PromptEnhancer` |
| **Judge** | Stage-1 LLM-as-judge scoring rollouts | `JUDGE_*` env (defaults to evolver) |

## Candidate representation

`rewrite_new_rollout_system_prompt.md` is the seed. The JSON output contract,
role, model-mechanics context, and the 6-segment task are **frozen** (outside
`<!-- EVOLVE-BLOCK-START/END -->`). Only the **behavioral policy blocks** inside
the markers are evolved. The evaluator strips the marker lines before sending
the prompt to the task model, so they never leak.

## Evaluation (Stage 1 only, for now)

Cheap proxy, **no video render**, so the loop iterates fast:

1. run the real dreamverse rewrite path with the candidate prompt over `eval_bank.jsonl`
2. **hard structural guard**: JSON ok, exactly 6 non-empty segments, no fallback → else score 0
3. frontier **LLM-judge** rubric: `seam_continuity` (.40), `instruction_follow` (.30),
   `variety`/anti-static (.20), `house_style` (.10)

Returned metrics: `combined_score`, `prompt_length` + `static_rate`
(MAP-Elites feature dims → a fast↔good / lively↔static Pareto grid),
plus `valid_rate`, per-axis means, `avg_task_latency_ms`. Structural failures
and weak seams are returned as **artifacts** and fed back to the evolver.

**Stage 2** (render + 5 boundary windows + human/video eval via `EvolveService`)
is a documented stub at the bottom of `evaluator.py`. Wire it, then flip
`cascade_evaluation: true` so only Stage-1 survivors pay the expensive cost.

## Run

This dir lives in FastVideo (`apps/dreamverse/prompt_evolution/`); the OpenEvolve
runtime is a separate dependency — clone https://github.com/codelion/openevolve
and `pip install -e .` it (provides `openevolve-run.py` + the `openevolve` package).

```bash
EXP=apps/dreamverse/prompt_evolution                       # from the FastVideo repo root
source "$EXP/env.local.sh"          # OPENAI_API_KEY (gitignored)
export CEREBRAS_API_KEY=...          # task model; or GROQ_API_KEY

# edit config.yaml: set llm.api_base + llm.models[0].name to your frontier model

# smoke test the evaluator on the seed (1-2 cases):
EVAL_BANK_LIMIT=2 python "$EXP/evaluator.py"

# full evolution run (openevolve-run.py from the openevolve clone):
python /path/to/openevolve/openevolve-run.py \
  "$EXP/rewrite_new_rollout_system_prompt.md" \
  "$EXP/evaluator.py" \
  --config "$EXP/config.yaml" \
  --iterations 50
```

## Files

- `rewrite_new_rollout_system_prompt.md` — seed prompt (contract frozen, policy blocks evolvable)
- `config.yaml` — evolver LLM, MAP-Elites, cascade settings
- `evaluator.py` — Stage-1 proxy (`evaluate_stage1` / `evaluate`) + Stage-2 TODO
- `eval_bank.jsonl` — fixed set of vibe-directing user ideas
- `env.local.sh` — secrets (gitignored)
