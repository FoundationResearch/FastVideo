"""boundary.audio — audio boundary (seam) artifact detector. Reasoning-free.

Per-(video-)frame audio loudness-discontinuity (|Δ RMS|) and spectral-flux, robust-z'd
vs the within-segment baseline. A seam is an AUDIO artifact if max audio z within a TIGHT
±win_s window at the join (default ±0.1s, right at the boundary) >= thresh — catching
silence gaps, abrupt onsets/pops, and hard audio cuts where the bed resets.

Inputs: sample["audio"] (path), sample["video"] (for n_frames / fps),
sample["auxiliary_info"]["seam_frames"]. score = audio artifact_rate.
"""
from __future__ import annotations
import numpy as np

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult
from fastvideo.eval.metrics.boundary._shared import default_seams, robust_z, window_max_abs_z, audio_per_frame


@register("boundary.audio")
class BoundaryAudioMetric(BaseMetric):
    name = "boundary.audio"
    dependencies = ["librosa", "soundfile"]

    def __init__(self, thresh: float = 4.0, win_s: float = 0.1) -> None:
        super().__init__()
        self.thresh, self.win_s = thresh, win_s

    def compute(self, sample: dict) -> MetricResult:
        audio = sample.get("audio")
        if audio is None:
            return self._skip(sample, "missing 'audio'")
        fps = float(sample.get("fps") or 24)
        video = sample.get("video")
        n = int(video.shape[0]) if video is not None else None
        seams = (sample.get("auxiliary_info") or {}).get("seam_frames")
        if n is None:
            # estimate frame count from seams or audio length
            import soundfile as sf
            info = sf.info(audio); n = int(round(info.duration * fps))
        seams = seams or default_seams(n)
        seams = [int(b) for b in seams if 0 < int(b) < n]
        win = max(1, int(round(self.win_s * fps)))
        seam_delta = [b - 1 for b in seams]
        try:
            disc, flux, tpr, rms, y, sr = audio_per_frame(audio, n, fps)
        except Exception as e:
            return self._skip(sample, f"audio read failed: {type(e).__name__}: {e}")
        if len(disc) == 0:
            return self._skip(sample, "audio too short")
        zd = robust_z(disc, seam_delta, win)
        zf = robust_z(flux, seam_delta, win)
        boundaries = []
        for b in seams:
            zda = window_max_abs_z(zd, b, win)
            zfa = window_max_abs_z(zf, b, win)
            amax = max(zda, zfa)
            if amax >= self.thresh:
                boundaries.append({"time_s": round(b / fps, 2), "frame": b, "z": round(amax, 1),
                                   "signal": "audio_disc" if zda >= zfa else "audio_flux",
                                   "context": {"audio_disc": round(zda, 1), "audio_flux": round(zfa, 1)}})
        nn = len(seams); sev = float(sum(x["z"] for x in boundaries))
        return MetricResult(name=self.name, score=round(len(boundaries) / max(1, nn), 4),
                            details={"boundaries": boundaries, "n_artifacts": len(boundaries), "n_seams": nn,
                                     "severity": round(sev, 1), "severity_per_seam": round(sev / max(1, nn), 3),
                                     "window_s": self.win_s, "seams": seams,
                                     "tonality_track_median": round(float(np.median(tpr)), 3),
                                     "zser": {"audio_disc": zd.tolist(), "audio_flux": zf.tolist()},
                                     "tonality_track": tpr.tolist(), "thresh": self.thresh})
