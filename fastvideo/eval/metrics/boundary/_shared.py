"""Shared helpers for boundary-artifact metrics (underscore -> not auto-imported).

Seam-aware anomaly detection for autoregressively-generated multi-segment video:
each ~Ns segment is conditioned only on the tail of the previous one, so artifacts
concentrate at the segment joins (seams). We score each seam as a robust z-anomaly
of a per-frame signal vs the within-segment baseline.

seam_frames = frame index of the FIRST frame of each continuation segment (the join).
"""
from __future__ import annotations
import numpy as np


def default_seams(n_frames: int) -> list[int]:
    """Dreamverse layout (seg1=120 frames, then 111 each) or an even 6-way split."""
    if 670 <= n_frames <= 680:
        return [120, 231, 342, 453, 564]
    k = 6
    return [round(n_frames * i / k) for i in range(1, k)]


def robust_z(series, seam_delta, exclude: int) -> np.ndarray:
    """Robust z (median/MAD) of a per-frame series vs the interior (non-seam) baseline."""
    s = np.asarray(series, float)
    s = np.nan_to_num(s, nan=float(np.nanmedian(s)) if len(s) else 0.0)
    mask = np.ones(len(s), bool)
    for k in seam_delta:
        for d in range(-exclude, exclude + 1):
            if 0 <= k + d < len(s):
                mask[k + d] = False
    interior = s[mask] if mask.any() else s
    med = float(np.median(interior))
    mad = float(np.median(np.abs(interior - med))) + 1e-9
    return (s - med) / (1.4826 * mad)


def exact_z(zlist, seam_frame: int, w: int = 2) -> float:
    """Max |z| within ±w frames of the join (delta index seam_frame-1)."""
    z = np.asarray(zlist, float)
    i = seam_frame - 1
    lo, hi = max(0, i - w), min(len(z), i + w + 1)
    seg = z[lo:hi]
    return float(seg[np.argmax(np.abs(seg))]) if len(seg) else 0.0


def window_max_abs_z(zlist, seam_frame: int, w: int) -> float:
    z = np.asarray(zlist, float)
    i = seam_frame - 1
    lo, hi = max(0, i - w), min(len(z), i + w + 1)
    seg = z[lo:hi]
    return float(np.max(np.abs(seg))) if len(seg) else 0.0


def audio_per_frame(audio_path: str, n_frames: int, fps: float):
    """Per-(video-)frame audio features from a wav/audio file.
    Returns (audio_disc[n-1], audio_flux[n-1], tonality[n] raw, rms[n], y, sr)."""
    import soundfile as sf
    import librosa
    y, sr = sf.read(audio_path)
    y = np.asarray(y, float)
    if y.ndim > 1:
        y = y.mean(axis=1)
    hop = max(1, int(round(sr / fps)))
    rms = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n_frames)])
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) + 1e-9
    P = S ** 2
    flux = np.sqrt((np.diff(S, axis=1) ** 2).sum(axis=0))
    k = max(1, P.shape[0] // 100)
    tpr = np.sort(P, axis=0)[::-1][:k, :].sum(axis=0) / (P.sum(axis=0) + 1e-12)

    def fit(a, L):
        a = np.asarray(a, float)
        return a[:L] if len(a) >= L else np.pad(a, (0, L - len(a)), mode="edge")

    disc = np.abs(np.diff(rms)) if n_frames > 1 else np.zeros(0)
    return disc, fit(flux, max(0, n_frames - 1)), fit(tpr, n_frames), rms, y, sr
