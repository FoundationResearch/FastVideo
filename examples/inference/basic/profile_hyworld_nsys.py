# SPDX-License-Identifier: Apache-2.0
"""
Minimal HYWorld inference script for Nsight Systems (nsys) profiling.

Run under nsys for fast, system-level profiling (no slow PyTorch profiler trace write):

  nsys profile -o hyworld_profile --trace=cuda,nvtx \\
    python examples/inference/basic/profile_hyworld_nsys.py

Or use the wrapper script:

  bash scripts/profile_nsys_hyworld.sh

Uses short clip (w-7, 29 frames) by default so the profile run finishes quickly.
"""

import sys
import os

# Ensure repo root and this example dir are on path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _d in (_REPO_ROOT, _THIS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import time
from basic_hyworld import (
    HYWorldVideoGenerator,
    DEFAULT_PROMPT,
    DEFAULT_IMAGE,
    get_resolution_from_image,
)


def main():
    # Short clip for quick profiling: w-7 -> 29 frames
    pose = "w-63"
    num_frames = 253
    resolution = "480p"
    output_path = "video_samples_hyworld_nsys_profile"
    HEIGHT, WIDTH = get_resolution_from_image(DEFAULT_IMAGE, resolution)

    print("Nsys profile target: HYWorld inference (short clip)")
    print(f"Pose: {pose}, num_frames: {num_frames}, resolution: {HEIGHT}x{WIDTH}")
    print(f"Output: {output_path}\n")

    # Disable PyTorch profiler so nsys is the only profiler
    os.environ.pop("FASTVIDEO_TORCH_PROFILER_DIR", None)

    print("Initializing VideoGenerator...")
    generator = HYWorldVideoGenerator.from_pretrained(
        "FastVideo/HY-WorldPlay-Bidirectional-Diffusers",
        num_gpus=4,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        image_encoder_cpu_offload=True,
    )

    print("Generating video (under nsys)...")
    start = time.perf_counter()
    generator.generate_video(
        prompt=DEFAULT_PROMPT,
        image_path=DEFAULT_IMAGE,
        output_path=output_path,
        save_video=True,
        negative_prompt="",
        num_frames=num_frames,
        fps=24,
        height=HEIGHT,
        width=WIDTH,
        seed=42,
        pose=pose,
        num_inference_steps=1,
        sigmas=None,
    )
    elapsed = time.perf_counter() - start
    print(f"\nDone in {elapsed:.2f}s. Trace: run nsys with -o <name> to get .nsys-rep")


if __name__ == "__main__":
    main()
