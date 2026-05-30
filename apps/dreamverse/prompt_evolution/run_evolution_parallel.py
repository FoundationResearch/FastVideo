"""Parallel evolution loop — fan a generation's candidates across multiple GPUs.

Same algorithm as run_evolution_demo (gpt-5.1 mutate -> gpt-oss-120b rewrite -> render
-> lean video metrics), but each generation's candidates are evaluated CONCURRENTLY:

  - mutate + rewrite (API calls) run in a thread pool;
  - render + score run in one gpu_worker.py subprocess per GPU (CUDA_VISIBLE_DEVICES
    pinned), each doing render-then-score internally so memory never co-locates.

With G GPUs a generation of N candidates costs ~ceil(N/G) * (render+score) instead of
N * (render+score). Writes viz/runs/<run>/evolution.json after each generation.

Run (single node, 4 GPUs):
  source env.local.sh
  export PYTHONPATH=$PWD LD_LIBRARY_PATH=/home/shared-bin/cuda-12.9/lib64:$LD_LIBRARY_PATH
  python run_evolution_parallel.py --generations 4 --per-gen 8 --gpus 0,1,2,3 --run-id chase_par
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_evolution_demo import (  # noqa: E402
    EVAL_IDEA, EVOLVER_MODEL, full_prompt, mutate, run_rollout, split_seed, validate_rollout, weakest_dim)

PY = sys.executable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def prepare_candidate(head, block, tail, cand_id, mut_summary, parent_id, gen, run_dir) -> dict:
    """API stage (threadable): rewrite the candidate's system prompt into 6 segments."""
    sysprompt = full_prompt(head, block, tail)
    rec = {
        "id": cand_id,
        "generation": gen,
        "parent_id": parent_id,
        "island": 0,
        "mutation_summary": mut_summary,
        "system_prompt": sysprompt,
        "policy_block": block,
        "prompt_length": len(sysprompt),
        "eval_idea": EVAL_IDEA,
        "out_mp4": os.path.join(run_dir, "videos", f"{cand_id}.mp4")
    }
    try:
        rw = run_rollout(sysprompt, EVAL_IDEA)
        ok, reason = validate_rollout(rw)
        rec["segments"] = list(rw.prompts) if ok else []
        if not ok:
            rec["error"] = reason
    except Exception as e:
        rec["segments"], rec["error"] = [], f"rewrite: {type(e).__name__}: {e}"
    return rec


def run_gpu_batch(gpu: int, items: list, work_path: str) -> list:
    """Launch one gpu_worker.py subprocess pinned to `gpu`; return updated items."""
    with open(work_path, "w") as f:
        json.dump(items, f)
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(gpu),
               METRIC_DEVICE="cuda:0",
               PYTHONPATH=REPO_ROOT,
               ENABLE_TORCH_COMPILE="0")
    subprocess.run([PY, os.path.join(HERE, "gpu_worker.py"), "--work", work_path], env=env, check=True)
    with open(work_path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--per-gen", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--run-id", default="par")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]

    run_dir = os.path.join(HERE, "viz", "runs", args.run_id)
    work_dir = os.path.join(run_dir, "_work")
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    head, seed_block, tail = split_seed()

    data = {
        "run_id": args.run_id,
        "target": "rewrite_new_rollout system prompt",
        "evolver_model": EVOLVER_MODEL,
        "task_model": "gpt-oss-120b",
        "feature_dimensions": ["prompt_length", "static_rate"],
        "eval_idea": EVAL_IDEA,
        "seed_policy_block": seed_block,
        "generations": []
    }

    def flush() -> None:
        with open(os.path.join(run_dir, "evolution.json"), "w") as f:
            json.dump(data, f, indent=2)

    flush()
    best_block, best_score, weakness = seed_block, -1e9, ""
    cand_n = 0
    for gen in range(args.generations):
        # 1) build this generation's policy blocks
        specs: list = []
        for k in range(args.per_gen):
            cid = f"candidate_{cand_n}"
            if gen == 0 and k == 0:
                specs.append((cid, seed_block, "seed prompt (baseline policy)", None))
            else:
                specs.append((cid, None, f"gpt-5.1 mutation of best (gen {gen})", f"best_gen{gen - 1}"))
            cand_n += 1

        # 2) mutate (gpt-5.1) + rewrite (cerebras) concurrently
        def _prep(spec, idx, best_block=best_block, weakness=weakness, gen=gen) -> dict:
            cid, block, mut, parent = spec
            if block is None:
                block = mutate(seed_block, best_block, weakness, idx)
            return prepare_candidate(head, block, tail, cid, mut, parent, gen, run_dir)

        print(f"[par] gen {gen}: preparing {len(specs)} candidates (mutate+rewrite) ...", flush=True)
        with ThreadPoolExecutor(max_workers=len(specs)) as ex:
            recs = list(ex.map(lambda p: _prep(p[1], p[0]), list(enumerate(specs))))

        # 3) split across GPUs, render+score each batch in parallel subprocesses
        valid = [r for r in recs if r.get("segments")]
        batches: dict = {g: [] for g in gpus}
        for i, r in enumerate(valid):
            batches[gpus[i % len(gpus)]].append(r)
        print(f"[par] gen {gen}: rendering+scoring {len(valid)} on {len(gpus)} GPUs ...", flush=True)
        with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
            done = list(
                ex.map(
                    lambda gb, gen=gen: run_gpu_batch(gb[0], gb[1], os.path.join(work_dir, f"g{gen}_gpu{gb[0]}.json")),
                    [(g, b) for g, b in batches.items() if b]))
        scored = {it["id"]: it for batch in done for it in batch}

        # 4) assemble generation records
        cands = []
        for r in recs:
            s = scored.get(r["id"], {})
            combined = s.get("combined", 0.0)
            r.update({
                "combined_score": combined,
                "video_score": combined,
                "metrics": s.get("metrics", {}),
                "video": f"videos/{r['id']}.mp4" if s.get("metrics") else None
            })
            cands.append({k: v for k, v in r.items() if k not in ("out_mp4", "rinfo", "seam_frames")})
        cands.sort(key=lambda c: -c["combined_score"])
        data["generations"].append({"generation": gen, "candidates": cands})
        flush()

        gen_best = cands[0]
        if gen_best["combined_score"] > best_score:
            best_score = gen_best["combined_score"]
            best_block = gen_best.get("policy_block", best_block)
            weakness = weakest_dim(gen_best.get("metrics", {}))
        print(f"[par] gen {gen} best={best_score:.3f}", flush=True)

    print(f"[par] done. best={best_score:.3f} -> viz run '{args.run_id}'", flush=True)


if __name__ == "__main__":
    main()
