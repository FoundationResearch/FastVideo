# Dreamverse rewriter-evolution: metrics guide (paste into OpenEvolve `prompt.system_message`)

You are improving a **prompt-rewriter system prompt** for Dreamverse. The rewriter turns
one high-level instruction into 6 segment prompts; the video is generated
**autoregressively** — each ~5s segment is conditioned only on the last few frames of the
previous one. So the characteristic failures happen **at the segment boundaries (seams)**:
the scene/subject resets, the camera trajectory flips, or the audio cuts/pops/whirs.

Your goal: evolve the rewriter prompt so the produced 6-segment videos have **few/no seam
artifacts and good overall quality**. Each candidate is scored by an objective,
reasoning-free evaluator over the videos it produces. Higher `combined_score` = better.

## Metrics (all per-video, reference-free)

**Quality — VBench (0–1, HIGHER = better):**
- `subject_consistency` — subject stays the same across frames (DINO features)
- `background_consistency` — background stays consistent (CLIP features)
- `temporal_flickering` — 1 − high-frequency jitter (higher = smoother)
- `aesthetic_quality`, `imaging_quality` — learned perceptual quality
- `vbench_mean` — average of the above

**Video boundary artifacts (objective detector, LOWER = better):**
- `video_artifact_rate` — fraction of seams with a content/identity **reset** (DINOv2
  feature jump *exactly at the join*, anomaly magnitude z ≥ 4 vs the clip's own
  within-segment baseline)
- `video_severity_per_seam` — mean magnitude of those jumps
- artifact `boundaries` list — exactly which seam time failed and the z magnitude

**Audio boundary artifacts (objective detector, LOWER = better):**
- `audio_artifact_rate` — fraction of seams with a sharp audio discontinuity (silence
  gap / pop / cut) in a tight **±0.1s** window at the join
- `audio_severity_per_seam` — mean magnitude
- `boundaries` list — which seam time, magnitude, and whether loudness (`audio_disc`) or
  spectral (`audio_flux`)

**Audio noise (clip-level, LOWER = better):**
- `audio_noise_score` — sustained tonal "whirring"/annoyance (high tonal-power-ratio +
  harmonic fraction + monotone loudness + low cleanliness)
- `squim_pesq` — learned speech-quality proxy (HIGHER = cleaner; ~1 = worst)

## How to act on it
- **Video artifacts high** → the rewriter is making segments discontinuous (new
  scene/subject/camera each segment). Push it to write each segment as a *continuation*
  of the previous (same subject, location, lighting, camera trajectory) unless a
  deliberate cut is genuinely wanted; avoid introducing new on-screen elements at segment
  starts.
- **Audio artifacts / noise high** → the rewriter injects per-segment audio cues
  ("the motor hums", "a siren wails", new sounds each segment) that reset the audio bed or
  create tonal machine whirring. Push it toward one *continuous ambient bed* across all
  segments and away from loud tonal motor/engine/whine sounds.
- The `boundaries` lists in the evaluator feedback tell you which seam times to target.

`combined_score = vbench_mean − 0.15·video_severity_per_seam − 0.15·audio_severity_per_seam − 0.10·audio_noise_score`
(weights are tunable; raise the audio/video weights to prioritize whichever you care about).
