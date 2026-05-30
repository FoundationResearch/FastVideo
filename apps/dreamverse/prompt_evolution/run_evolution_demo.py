"""
Real small-scale evolution run that produces live visualization data.

Orchestrates the actual components (same as the OpenEvolve harness, just a simple
loop instead of the full MAP-Elites controller, so it's fast/reliable for a demo):

  evolver (gpt-5.1) mutates the policy block  ->  rewrite (gpt-oss-120b @ Cerebras)
  produces 6 segments  ->  render 30s mp4 (persistent LTX2 worker)  ->  video_scorer.

Writes viz/runs/<run_id>/evolution.json + videos/ INCREMENTALLY (after every
candidate) so the dashboard at alexzms3.ngrok.app populates live.

Run (in the fixed alexfvi env, after `source env.local.sh`):
    python run_evolution_demo.py --generations 3 --per-gen 3 --run-id live

The candidate = the mutable policy region of the seed prompt (between the
<!-- EVOLVE-BLOCK --> markers); the frozen contract head/tail are reattached for
the task model.
"""

import argparse
import json
import os
import sys

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluator import run_rollout, validate_rollout  # noqa: E402
from render_rollout import RolloutRenderer  # noqa: E402
from video_scorer import score_rollout  # noqa: E402

SEED_MD = os.path.join(HERE, "rewrite_new_rollout_system_prompt.md")
EVAL_IDEA = os.environ.get("EVAL_IDEA", "two roommates argue over the last cup of coffee in a tiny kitchen")

EVOLVER_MODEL = os.environ.get("EVOLVER_MODEL", "gpt-5.1")
_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EVOLVER_SYSTEM = """You are an expert prompt engineer optimizing the BEHAVIORAL POLICY
of a real-time video "rewriter" LLM. Given a one-line user idea, that rewriter outputs
a JSON rollout of exactly 6 sequential 5-second segment prompts for the ltx2 video model
(30s total), each segment conditioned only on the last frames/audio of the previous one.

You are given the current policy text and (if any) the measured weaknesses of the videos
it produced. Revise the policy to improve seam continuity, faithful coverage of the user's
idea, vivid grounded staging, and LIVELY non-static motion (avoid static wallpaper).

Keep it a drop-in replacement for the policy region: same tag structure, no JSON contract,
no preamble. Return ONLY the revised policy text."""


def split_seed() -> tuple[str, str, str]:
    """Return (head, evolve_block, tail) of the seed md, by EVOLVE-BLOCK markers."""
    with open(SEED_MD) as f:
        text = f.read()
    a = text.index("<!-- EVOLVE-BLOCK-START -->")
    b = text.index("<!-- EVOLVE-BLOCK-END -->")
    head = text[:a]
    block = text[a + len("<!-- EVOLVE-BLOCK-START -->"):b].strip("\n")
    tail = text[b + len("<!-- EVOLVE-BLOCK-END -->"):]
    return head, block, tail


def full_prompt(head: str, block: str, tail: str) -> str:
    return (head + "\n" + block + "\n" + tail).strip()


def mutate(seed_block: str, best_block: str, weakness: str, idx: int) -> str:
    """Ask the evolver LLM for a revised policy block."""
    user = (f"SEED POLICY (reference):\n{seed_block}\n\n"
            f"CURRENT BEST POLICY:\n{best_block}\n\n"
            f"MEASURED WEAKNESSES: {weakness or 'none yet'}\n\n"
            f"Produce revision #{idx}: a distinct improvement. Return only the policy text.")
    resp = _client.chat.completions.create(
        model=EVOLVER_MODEL,
        messages=[{
            "role": "system",
            "content": EVOLVER_SYSTEM
        }, {
            "role": "user",
            "content": user
        }],
        temperature=0.9,
    )
    return resp.choices[0].message.content.strip()


def weakest_dim(metrics: dict) -> str:
    dims = {k: v for k, v in metrics.items() if k not in ("video_score", "clip_used", "num_frames", "num_segments")}
    if not dims:
        return ""
    k = min(dims, key=lambda x: dims[x])
    return f"lowest dimension was {k}={dims[k]:.2f}"


def evaluate_candidate(head: str, block: str, tail: str, renderer, run_dir: str, cand_id: str) -> dict:
    """Full pipeline for one candidate: rewrite -> render -> score. Returns record."""
    sysprompt = full_prompt(head, block, tail)
    rewrite = run_rollout(sysprompt, EVAL_IDEA)
    ok, reason = validate_rollout(rewrite)
    if not ok:
        return {
            "video_score": 0.0,
            "metrics": {},
            "video": None,
            "error": reason,
            "segments": [],
            "system_prompt": sysprompt,
            "policy_block": block
        }
    segments = list(rewrite.prompts)
    out_mp4 = os.path.join(run_dir, "videos", f"{cand_id}.mp4")
    rinfo = renderer.render(segments, out_mp4)

    if os.environ.get("METRIC_MODE", "lean") == "lean":
        import stage2_lean
        combined, metrics, _ = stage2_lean.score(out_mp4, rinfo, segments, EVAL_IDEA)
    else:  # legacy lightweight scorer
        s = score_rollout(out_mp4, segment_prompts=segments, segment_boundaries=rinfo["segment_frame_counts"])
        combined = s["video_score"]
        metrics = {k: v for k, v in s.items() if k not in ("video_score", "clip_used", "num_frames", "num_segments")}
    return {
        "video_score": combined,
        "metrics": metrics,
        "video": f"videos/{cand_id}.mp4",
        "segments": segments,  # the 6 rewritten segment prompts
        "system_prompt": sysprompt,  # full prompt fed to the task model
        "policy_block": block,  # the evolved (mutable) region only
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--per-gen", type=int, default=3)
    ap.add_argument("--run-id", default="live")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    run_dir = os.path.join(HERE, "viz", "runs", args.run_id)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    head, seed_block, tail = split_seed()

    data = {
        "run_id": args.run_id,
        "target": "rewrite_new_rollout system prompt",
        "evolver_model": EVOLVER_MODEL,
        "task_model": "gpt-oss-120b",
        "feature_dimensions": ["prompt_length", "static_rate"],
        "eval_idea": EVAL_IDEA,
        "generations": [],
    }

    def flush() -> None:
        with open(os.path.join(run_dir, "evolution.json"), "w") as f:
            json.dump(data, f, indent=2)

    flush()
    print(f"[evolve] loading render worker on gpu {args.gpu} ...", flush=True)
    renderer = RolloutRenderer(gpu_id=args.gpu)
    print(f"[evolve] worker ready ({renderer.init_s}s)", flush=True)

    best_block, best_score, weakness = seed_block, -1.0, ""
    cand_n = 0
    try:
        for gen in range(args.generations):
            cands: list = []
            for k in range(args.per_gen):
                cid = f"candidate_{cand_n}"
                if gen == 0 and k == 0:
                    block, mut = seed_block, "seed prompt (baseline policy)"
                else:
                    block = mutate(seed_block, best_block, weakness, cand_n)
                    mut = f"gpt-5.1 mutation of best (gen {gen})"
                print(f"[evolve] gen {gen} {cid}: rewrite+render+score ...", flush=True)
                rec = evaluate_candidate(head, block, tail, renderer, run_dir, cid)
                rec.update({
                    "id": cid,
                    "generation": gen,
                    "parent_id": (None if gen == 0 and k == 0 else f"best_gen{gen - 1}"),
                    "island": 0,
                    "combined_score": rec["video_score"],
                    "prompt_length": len(full_prompt(head, block, tail)),
                    "mutation_summary": mut,
                    "prompt_excerpt": block[:240].replace("\n", " "),
                })
                cands.append({"block": block, **rec})
                cand_n += 1
                # incremental flush so the dashboard grows live
                data["generations"] = _materialize(data, gen, cands)
                flush()
                print(f"[evolve]   -> score {rec['combined_score']:.3f}", flush=True)

            gen_best = max(cands, key=lambda c: c["combined_score"])
            if gen_best["combined_score"] > best_score:
                best_score, best_block = gen_best["combined_score"], gen_best["block"]
                weakness = weakest_dim(gen_best.get("metrics", {}))
            print(f"[evolve] gen {gen} best={best_score:.3f}", flush=True)
    finally:
        renderer.close()
        flush()
    print(f"[evolve] done. best={best_score:.3f}  ->  viz run '{args.run_id}'", flush=True)


def _materialize(data, upto_gen, current_cands) -> list:
    """Rebuild generations list keeping prior gens + the in-progress one."""
    gens = [g for g in data["generations"] if g["generation"] < upto_gen]
    cur = [{k: v for k, v in c.items() if k != "block"} for c in current_cands]
    cur.sort(key=lambda c: -c["combined_score"])
    gens.append({"generation": upto_gen, "candidates": cur})
    return gens


if __name__ == "__main__":
    main()
