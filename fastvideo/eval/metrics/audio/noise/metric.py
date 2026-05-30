"""audio.noise — clip-level tonal-noise / "whirring" annoyance (NOT boundary-specific).

No-reference. A constant tonal whine (e.g. motor/rotor whirring) is sustained, so it is
NOT a seam discontinuity — it shows as concentrated tonal energy held across the clip.
We quantify it from: tonal-power-ratio (top-1% spectral bins / total), harmonic fraction
(HPSS), loudness monotony (low rms CV), and optional torchaudio-SQUIM cleanliness (PESQ).

score = noise_score (higher = noisier/more annoying). Reference-free, per clip.
"""
from __future__ import annotations
import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult
from fastvideo.eval.metrics.boundary._shared import audio_per_frame


@register("audio.noise")
class AudioNoiseMetric(BaseMetric):
    name = "audio.noise"
    dependencies = ["librosa", "soundfile"]

    def __init__(self, use_squim: bool = True) -> None:
        super().__init__()
        self.use_squim = use_squim
        self._squim = None
        self._squim_sr = None

    def setup(self) -> None:
        if self.use_squim and self._squim is None:
            try:
                from torchaudio.pipelines import SQUIM_OBJECTIVE
                self._squim = SQUIM_OBJECTIVE.get_model().to(self.device).eval()
                self._squim_sr = SQUIM_OBJECTIVE.sample_rate
            except Exception:
                self._squim = None

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        audio = sample.get("audio")
        if audio is None:
            return self._skip(sample, "missing 'audio'")
        fps = float(sample.get("fps") or 24)
        video = sample.get("video")
        import soundfile as sf
        n = int(video.shape[0]) if video is not None else int(round(sf.info(audio).duration * fps))
        try:
            import librosa
            disc, flux, tpr, rms, y, sr = audio_per_frame(audio, n, fps)
        except Exception as e:
            return self._skip(sample, f"audio read failed: {type(e).__name__}: {e}")
        tonal_power_ratio = float(np.median(tpr))
        harm, _ = librosa.effects.hpss(y)
        harm_frac = float(np.sum(harm ** 2) / (np.sum(y ** 2) + 1e-12))
        rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-9))
        details = {"tonal_power_ratio": round(tonal_power_ratio, 3),
                   "harmonic_fraction": round(harm_frac, 3), "rms_cv": round(rms_cv, 3)}
        noise = tonal_power_ratio + harm_frac + max(0.0, 1.0 - rms_cv)
        if self.use_squim:
            if self._squim is None:
                self.setup()
            if self._squim is not None:
                try:
                    import torchaudio
                    # feed SQUIM the waveform we already loaded (avoid torchaudio.load/torchcodec)
                    wav = torch.from_numpy(np.asarray(y, dtype="float32"))[None]
                    if sr != self._squim_sr:
                        wav = torchaudio.functional.resample(wav, sr, self._squim_sr)
                    stoi, pesq, sisdr = self._squim(wav.to(self.device))
                    details["squim_pesq"] = round(float(pesq), 3)
                    details["squim_stoi"] = round(float(stoi), 3)
                    noise += max(0.0, (2.5 - float(pesq)) / 2.5)
                except Exception as e:
                    details["squim_error"] = str(e)[:80]
        details["noise_score"] = round(noise, 3)
        return MetricResult(name=self.name, score=round(noise, 3), details=details)
