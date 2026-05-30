"""OpenEvolve evaluator for a dreamverse rollout — uses the REGISTERED fastvideo.eval
metrics (VBench + boundary.video + boundary.audio + audio.noise) in a single
create_evaluator call, and assembles the OpenEvolve result:

  { "combined_score": float,          # fitness, HIGHER = better
    "metrics": {...named scalars...},  # sub-objectives / MAP-Elites features
    "artifacts": { "metrics_guide": ..., "video_artifacts": ..., "audio_artifacts": ..., "noise": ... } }

CLI:
  python openevolve_eval.py <video.mp4> [--seams 120,231,342,453,564] [--prompt "..."] [--out f.json] [--no-vbench]
Run in a venv that has the eval extras + LD_LIBRARY_PATH=/home/shared-bin/cuda-12.9/lib64.
"""
import argparse, json, os, subprocess
from fastvideo.eval import create_evaluator, samples_from

FFMPEG = os.environ.get("FASTVIDEO_FFMPEG_BIN", "/home/hal-shao/miniforge3/bin/ffmpeg")
VBENCH = ["vbench.subject_consistency", "vbench.background_consistency", "vbench.temporal_flickering",
          "vbench.aesthetic_quality", "vbench.imaging_quality"]
OURS = ["boundary.video", "boundary.audio", "audio.noise"]
_GUIDE = os.path.join(os.path.dirname(__file__), "METRICS_GUIDE.md")
METRICS_GUIDE = open(_GUIDE).read() if os.path.exists(_GUIDE) else "(see METRICS_GUIDE.md)"


def evaluate(video_path, seams=None, prompt="", run_vbench=True, device="cuda:0"):
    wav = os.path.splitext(video_path)[0] + ".eval.wav"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", video_path, "-ac", "1", "-ar", "48000", wav], check=True)
    metric_names = (VBENCH if run_vbench else []) + OURS
    ev = create_evaluator(metrics=metric_names, device=device)
    aux = {"seam_frames": seams} if seams else None
    res = ev.evaluate(samples=samples_from(video=video_path, audio=wav, text_prompt=prompt,
                                           fps=24.0, auxiliary_info=aux))[0]

    def det(name):
        r = res.get(name); return (getattr(r, "details", {}) or {}) if r else {}
    def sc(name):
        r = res.get(name); return getattr(r, "score", None) if r else None

    bv, ba, an = det("boundary.video"), det("boundary.audio"), det("audio.noise")
    metrics = {
        "video_artifact_rate": sc("boundary.video"), "video_severity_per_seam": bv.get("severity_per_seam"),
        "audio_artifact_rate": sc("boundary.audio"), "audio_severity_per_seam": ba.get("severity_per_seam"),
        "n_video_artifacts": bv.get("n_artifacts"), "n_audio_artifacts": ba.get("n_artifacts"),
        "n_seams": bv.get("n_seams") or ba.get("n_seams"),
        "audio_noise_score": sc("audio.noise"), "squim_pesq": an.get("squim_pesq"),
    }
    vbench_mean = None
    if run_vbench:
        vq = {m.split(".")[1]: sc(m) for m in VBENCH if sc(m) is not None}
        metrics.update(vq)
        if vq: vbench_mean = sum(vq.values()) / len(vq)
    metrics["vbench_mean"] = round(vbench_mean, 4) if vbench_mean is not None else None

    q = vbench_mean if vbench_mean is not None else 0.7
    vsev = metrics["video_severity_per_seam"] or 0.0
    asev = metrics["audio_severity_per_seam"] or 0.0
    noise = metrics["audio_noise_score"] or 0.0
    metrics["combined_score"] = round(q - 0.15 * vsev - 0.15 * asev - 0.10 * noise, 4)

    def fmt(d):
        b = d.get("boundaries") or []
        return "none" if not b else "; ".join(f"{x['time_s']}s z={x['z']}({x['signal']})" for x in b)
    artifacts = {
        "metrics_guide": METRICS_GUIDE,
        "video_artifacts": f"{bv.get('n_artifacts',0)}/{bv.get('n_seams','?')} seams: {fmt(bv)}",
        "audio_artifacts": f"{ba.get('n_artifacts',0)}/{ba.get('n_seams','?')} seams (±{ba.get('window_s','?')}s): {fmt(ba)}",
        "noise": f"noise_score={an.get('noise_score')} tonal={an.get('tonal_power_ratio')} "
                 f"harmonic={an.get('harmonic_fraction')} pesq={an.get('squim_pesq')}",
    }
    return {"combined_score": metrics["combined_score"], "metrics": metrics, "artifacts": artifacts,
            "video": video_path, "prompt": prompt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video"); p.add_argument("--seams", default=None); p.add_argument("--prompt", default="")
    p.add_argument("--out", default=None); p.add_argument("--no-vbench", action="store_true")
    a = p.parse_args()
    seams = [int(x) for x in a.seams.split(",")] if a.seams else None
    r = evaluate(a.video, seams, a.prompt, run_vbench=not a.no_vbench)
    out = a.out or (os.path.splitext(a.video)[0] + ".openevolve.json")
    json.dump(r, open(out, "w"), indent=2)
    print("combined_score:", r["combined_score"])
    print("metrics:", json.dumps(r["metrics"], indent=2))
    for k, v in r["artifacts"].items():
        if k != "metrics_guide": print(f"  {k}: {v}")
    print("[saved]", out)


if __name__ == "__main__":
    main()
