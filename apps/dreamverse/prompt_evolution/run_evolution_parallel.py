"""Parallel, MULTI-PROMPT evolution loop across GPUs.

Each candidate's fitness is the AVERAGE over a fixed set of eval prompts (so the system
prompt generalizes instead of overfitting one idea). Per candidate x eval-prompt we
rewrite -> render -> score; the candidate's combined_score / metrics are the mean over
its eval prompts. The viz shows ONE showcase prompt's video as the example.

Per generation: mutate (gpt-5.5) the best policy -> N candidate policy blocks; rewrite
every (candidate, eval-prompt) pair (threaded API); render+score all work items across
GPUs (one gpu_worker subprocess per GPU, render-then-score so memory never co-locates);
average per candidate; keep the global best as the next parent.

Run (single node, 4 GPUs):
  source env.local.sh
  export PYTHONPATH=$PWD LD_LIBRARY_PATH=/home/shared-bin/cuda-12.9/lib64:$LD_LIBRARY_PATH
  python run_evolution_parallel.py --generations 5 --per-gen 6 --gpus 0,1,2,3 \
      --eval-ids chase_alley,argue_makeup,creator_night,fpv_mtb,cat_swim --run-id search1
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_evolution_demo as _red  # noqa: E402

EVOLVER_MODEL = _red.EVOLVER_MODEL
full_prompt, mutate, run_rollout = _red.full_prompt, _red.mutate, _red.run_rollout
split_seed, validate_rollout, weakest_dim = _red.split_seed, _red.validate_rollout, _red.weakest_dim

PY = sys.executable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
BANK = os.path.join(HERE, "eval_bank.jsonl")


def load_eval_set(ids) -> list:
    """ids = list of bank ids, or the string 'all' for the whole bank (file order)."""
    order, bank = [], {}
    with open(BANK, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                bank[d["id"]] = d["prompt"]
                order.append(d["id"])
    if ids == "all":
        ids = order
    return [{"id": i, "prompt": bank[i]} for i in ids if i in bank]


def build_slots(use_slurm: bool, gpus_per_node: int, local_gpus: list) -> list:
    """Return list of (jobid_or_None, gpu). Local: jobid None. Slurm: one entry per GPU
    of every running single-node allocation the user holds (srun --jobid into each)."""
    if not use_slurm:
        return [(None, g) for g in local_gpus]
    import getpass
    out = subprocess.check_output(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i %T %D"], text=True)
    slots = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "RUNNING" and parts[2] == "1":
            for g in range(gpus_per_node):
                slots.append((parts[0], g))
    return slots


def run_gpu_batch(slot: tuple, items: list, work_path: str) -> list:
    """Render+score one batch on a GPU slot. work_path must be on shared (NFS) storage."""
    jobid, gpu = slot
    with open(work_path, "w") as f:
        json.dump(items, f, ensure_ascii=False)
    worker = os.path.join(HERE, "gpu_worker.py")
    if jobid is None:  # local subprocess
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=str(gpu),
                   METRIC_DEVICE="cuda:0",
                   PYTHONPATH=REPO_ROOT,
                   ENABLE_TORCH_COMPILE="0")
        subprocess.run([PY, worker, "--work", work_path], env=env, check=True)
    else:  # remote node via srun into the existing allocation (wrapper avoids quoting issues)
        wrapper = os.path.join(HERE, "gpu_worker_remote.sh")
        subprocess.run(
            ["srun", f"--jobid={jobid}", "--overlap", "--nodes=1", "--ntasks=1", "bash", wrapper, work_path,
             str(gpu)],
            check=True)
    with open(work_path) as f:
        return json.load(f)


def _mean_metrics(dicts: list) -> dict:
    keys: set = set()
    for d in dicts:
        keys |= set(d or {})
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts if d and isinstance(d.get(k), int | float)]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--per-gen", type=int, default=6)
    ap.add_argument("--gpus", default="0,1,2,3", help="local GPU ids (single-node mode)")
    ap.add_argument("--slurm", action="store_true", help="fan out across ALL your running slurm allocations")
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--eval-ids", default="chase_alley,argue_makeup,creator_night,fpv_mtb,cat_swim")
    ap.add_argument("--run-id", default="search1")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    slots = build_slots(args.slurm, args.gpus_per_node, gpus)
    eval_set = load_eval_set("all" if args.eval_ids.strip() == "all" else [s.strip() for s in args.eval_ids.split(",")])
    showcase = eval_set[0]["id"]

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
        "eval_idea": f"avg over {len(eval_set)} prompts: " + ", ".join(e["id"] for e in eval_set),
        "showcase_prompt": next(e["prompt"] for e in eval_set if e["id"] == showcase),
        "seed_policy_block": seed_block,
        "generations": []
    }

    def flush() -> None:
        with open(os.path.join(run_dir, "evolution.json"), "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    flush()
    best_block, best_score, weakness = seed_block, -1e9, ""
    cand_n = 0
    for gen in range(args.generations):
        # 1) candidate policy blocks
        specs: list = []
        for k in range(args.per_gen):
            cid = f"candidate_{cand_n}"
            specs.append((cid, seed_block if (gen == 0 and k == 0) else None, "seed prompt (baseline policy)" if
                          (gen == 0 and k == 0) else f"gpt-5.5 mutation of best (gen {gen})", None if
                          (gen == 0 and k == 0) else f"best_gen{gen - 1}"))
            cand_n += 1

        def _block(spec, idx, best_block=best_block, weakness=weakness) -> tuple:
            cid, block, mut, parent = spec
            if block is None:
                block = mutate(seed_block, best_block, weakness, idx)
            return cid, block, mut, parent

        print(f"[par] gen {gen}: mutating {len(specs)} policies (gpt-5.5) ...", flush=True)
        with ThreadPoolExecutor(max_workers=len(specs)) as ex:
            blocks = list(ex.map(lambda p: _block(p[1], p[0]), list(enumerate(specs))))

        # 2) build (candidate x eval-prompt) work items via threaded rewrite
        cand_meta = {}
        for cid, block, mut, parent in blocks:
            sp = full_prompt(head, block, tail)
            cand_meta[cid] = {
                "id": cid,
                "generation": gen,
                "parent_id": parent,
                "island": 0,
                "mutation_summary": mut,
                "system_prompt": sp,
                "policy_block": block,
                "prompt_length": len(sp)
            }
        pairs = [(cid, block, e) for cid, block, _, _ in blocks for e in eval_set]

        def _rewrite(pair, cand_meta=cand_meta) -> dict:
            cid, block, e = pair
            sp = cand_meta[cid]["system_prompt"]
            w = {
                "wid": f"{cid}__{e['id']}",
                "cand_id": cid,
                "prompt_id": e["id"],
                "eval_idea": e["prompt"],
                "out_mp4": os.path.join(run_dir, "videos", f"{cid}__{e['id']}.mp4")
            }
            try:
                rw = run_rollout(sp, e["prompt"])
                ok, reason = validate_rollout(rw)
                w["segments"] = list(rw.prompts) if ok else []
                if not ok:
                    w["error"] = reason
            except Exception as ex2:
                w["segments"], w["error"] = [], f"rewrite: {ex2}"
            return w

        print(f"[par] gen {gen}: rewriting {len(pairs)} (candidate x prompt) pairs ...", flush=True)
        with ThreadPoolExecutor(max_workers=min(16, len(pairs))) as ex:
            work = list(ex.map(_rewrite, pairs))

        # 3) render+score all work items across GPUs
        valid = [w for w in work if w.get("segments")]
        sb: list = [[] for _ in slots]
        for i, w in enumerate(valid):
            sb[i % len(slots)].append(w)
        active = [(idx, slots[idx], b) for idx, b in enumerate(sb) if b]
        print(f"[par] gen {gen}: rendering+scoring {len(valid)} items on {len(active)} GPU slots ...", flush=True)
        with ThreadPoolExecutor(max_workers=len(active)) as ex:
            done = list(
                ex.map(lambda t, gen=gen: run_gpu_batch(t[1], t[2], os.path.join(work_dir, f"g{gen}_slot{t[0]}.json")),
                       active))
        scored = {w["wid"]: w for batch in done for w in batch}

        # 4) average per candidate
        cands = []
        for cid, meta in cand_meta.items():
            items = [scored.get(f"{cid}__{e['id']}", {}) for e in eval_set]
            combineds = [it.get("combined", 0.0) for it in items]
            combined = round(sum(combineds) / len(combineds), 4) if combineds else 0.0
            show = scored.get(f"{cid}__{showcase}", {})
            rec = dict(meta)
            rec.update({
                "combined_score": combined,
                "video_score": combined,
                "metrics": _mean_metrics([it.get("metrics", {}) for it in items]),
                "video": f"videos/{cid}__{showcase}.mp4" if show.get("metrics") else None,
                "segments": show.get("segments", []),
                "eval_breakdown": {
                    e["id"]: round(scored.get(f"{cid}__{e['id']}", {}).get("combined", 0.0), 3)
                    for e in eval_set
                },
            })
            cands.append(rec)
        cands.sort(key=lambda c: -c["combined_score"])
        data["generations"].append({"generation": gen, "candidates": cands})
        flush()

        gen_best = cands[0]
        if gen_best["combined_score"] > best_score:
            best_score = gen_best["combined_score"]
            best_block = gen_best.get("policy_block", best_block)
            weakness = weakest_dim(gen_best.get("metrics", {}))
        print(f"[par] gen {gen} best_avg={best_score:.3f}", flush=True)

    print(f"[par] done. best_avg={best_score:.3f} -> viz run '{args.run_id}'", flush=True)


if __name__ == "__main__":
    main()
