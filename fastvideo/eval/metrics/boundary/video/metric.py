"""boundary.video — video boundary (seam) artifact detector for autoregressive
multi-segment video. Reasoning-free, magnitude-only.

Per-frame signals on the decoded video: DINO content-feature delta (the content/identity
RESET signal), RAFT global optical-flow magnitude + direction-change (camera motion
discontinuity / reversal), LPIPS perceptual delta, pixel MAD. Each is robust-z'd vs the
within-segment baseline. A seam is a VIDEO artifact if |DINO exact-z| (max |z| within
±exact_w frames of the join) >= thresh. Other signals are reported as context.

Seam positions come from sample["auxiliary_info"]["seam_frames"] (frame index of each
continuation segment's first frame); falls back to the dreamverse layout.

score = artifact_rate (fraction of seams flagged); details.boundaries = the flagged seams.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, normalize

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult
from fastvideo.eval.metrics.boundary._shared import default_seams, robust_z, exact_z

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@register("boundary.video")
class BoundaryVideoMetric(BaseMetric):
    name = "boundary.video"
    dependencies = ["lpips"]   # torchvision (RAFT) + torch.hub DINO are core

    def __init__(self, thresh: float = 4.0, exact_w: int = 2, win_s: float = 1.0,
                 flow_size=(288, 512), chunk: int = 32) -> None:
        super().__init__()
        self.thresh, self.exact_w, self.win_s = thresh, exact_w, win_s
        self.flow_size, self.chunk = flow_size, chunk
        self._dino = self._raft = self._raft_tf = self._lpips = None

    def to(self, device):
        super().to(device)
        for m in (self._dino, self._raft, self._lpips):
            if m is not None:
                m.to(self.device)
        return self

    def setup(self) -> None:
        if self._dino is not None:
            return
        self._dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(self.device).eval()
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        w = Raft_Large_Weights.DEFAULT
        self._raft = raft_large(weights=w, progress=False).to(self.device).eval()
        self._raft_tf = w.transforms()
        import lpips as _lpips
        self._lpips = _lpips.LPIPS(net="alex").to(self.device).eval()

    @torch.no_grad()
    def _signals(self, video: torch.Tensor) -> dict:
        # video: (T, C, H, W) float [0,1] on cpu
        T = video.shape[0]
        v = video.to(self.device)
        # --- DINO consecutive cosine distance ---
        d = normalize(resize(v, 224, antialias=False), mean=_MEAN, std=_STD)
        feats = []
        for i in range(0, T, self.chunk):
            feats.append(F.normalize(self._dino(d[i:i + self.chunk]).float(), dim=-1))
        feats = torch.cat(feats, 0)
        dino = (1 - (feats[:-1] * feats[1:]).sum(-1)).cpu().numpy().tolist()
        # --- small frames for flow/lpips/pixel ---
        small = resize(v, list(self.flow_size), antialias=True)          # (T,3,h,w) [0,1]
        sm255 = (small * 255.0)
        pix = (sm255[1:] - sm255[:-1]).abs().mean(dim=(1, 2, 3)).cpu().numpy().tolist()
        lp = (small * 2 - 1)
        lpips_v = []
        for t in range(T - 1):
            lpips_v.append(float(self._lpips(lp[t:t+1], lp[t+1:t+2]).item()))
        # --- RAFT global flow ---
        fmag = [0.0] * (T - 1); fdx = [0.0] * (T - 1); fdy = [0.0] * (T - 1)
        for t in range(T - 1):
            a, b = self._raft_tf(small[t:t+1], small[t+1:t+2])
            fl = self._raft(a, b)[-1][0]
            fdx[t] = float(fl[0].mean()); fdy[t] = float(fl[1].mean())
            fmag[t] = float((fl[0]**2 + fl[1]**2).sqrt().mean())
        ang = [math.atan2(fdy[t], fdx[t]) for t in range(T - 1)]
        fdir = [0.0] + [abs((ang[t] - ang[t-1] + math.pi) % (2 * math.pi) - math.pi) for t in range(1, T - 1)]
        return {"dino": dino, "flow_mag": fmag, "flow_dir": fdir, "lpips": lpips_v, "pixel": pix}

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        if self._dino is None:
            self.setup()
        video = sample.get("video")
        if video is None:
            return self._skip(sample, "missing 'video'")
        T = int(video.shape[0]); fps = float(sample.get("fps") or 24)
        seams = (sample.get("auxiliary_info") or {}).get("seam_frames") or default_seams(T)
        seams = [int(b) for b in seams if 0 < int(b) < T]
        win = max(1, int(round(self.win_s * fps)))
        seam_delta = [b - 1 for b in seams]
        sig = self._signals(video)
        zser = {k: robust_z(v, seam_delta, win).tolist() for k, v in sig.items()}
        boundaries = []
        for b in seams:
            dz = exact_z(zser["dino"], b, self.exact_w)
            if abs(dz) >= self.thresh:
                ctx = {k: round(exact_z(zser[k], b, self.exact_w), 1) for k in ("flow_mag", "flow_dir", "lpips", "pixel")}
                boundaries.append({"time_s": round(b / fps, 2), "frame": b, "z": round(abs(dz), 1),
                                   "signal": "dino", "context": ctx})
        n = len(seams); sev = float(sum(x["z"] for x in boundaries))
        return MetricResult(name=self.name, score=round(len(boundaries) / max(1, n), 4),
                            details={"boundaries": boundaries, "n_artifacts": len(boundaries), "n_seams": n,
                                     "severity": round(sev, 1), "severity_per_seam": round(sev / max(1, n), 3),
                                     "seams": seams, "zser": zser, "thresh": self.thresh})
