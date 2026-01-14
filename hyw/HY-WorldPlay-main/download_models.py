#!/usr/bin/env python3
"""
Download script for HY-WorldPlay models.
Downloads all required models from HuggingFace and ModelScope.

Usage:
    python download_models.py --hf_token <your_token>

The HF token is required for downloading the vision encoder from FLUX.1-Redux-dev.
Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
"""

import argparse
import os
import shutil
import sys


def _ensure_dir(p: str) -> str:
    p = os.path.expanduser(p)
    os.makedirs(p, exist_ok=True)
    return p


def _repo_local_dir(weights_root: str, repo_id: str) -> str:
    """
    Map a HF repo_id like 'tencent/HY-WorldPlay' into:
      <weights_root>/tencent/HY-WorldPlay
    """
    weights_root = os.path.expanduser(weights_root)
    local_dir = os.path.join(weights_root, *repo_id.split("/"))
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def _hf_snapshot_download(
    repo_id: str,
    *,
    weights_root: str,
    hf_cache_dir: str,
    allow_patterns=None,
    token=None,
):
    """
    Download a (possibly filtered) snapshot into a deterministic local_dir under weights_root.
    """
    from huggingface_hub import snapshot_download

    local_dir = _repo_local_dir(weights_root, repo_id)
    return snapshot_download(
        repo_id,
        allow_patterns=allow_patterns,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # make the weights folder self-contained
        cache_dir=hf_cache_dir,
        token=token,
    )


def check_dependencies():
    """Check and install required dependencies."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install -U 'huggingface_hub[cli]'")

    try:
        import modelscope
    except ImportError:
        print("Installing modelscope...")
        os.system("pip install modelscope")


def download_hy_worldplay(weights_root: str, hf_cache_dir: str):
    """Download HY-WorldPlay action models."""
    print("\n" + "=" * 60)
    print("[1/6] Downloading tencent/HY-WorldPlay...")
    print("=" * 60)

    # NOTE:
    # The upstream repo may contain references to files that 404 (e.g. wan_distilled_model/model.pt).
    # For training/inference in this project, we only need the action model safetensors.
    # So we download a minimal subset to avoid failing on unrelated files.
    worldplay_path = _hf_snapshot_download(
        "tencent/HY-WorldPlay",
        weights_root=weights_root,
        hf_cache_dir=hf_cache_dir,
        allow_patterns=[
            "ar_model/*",
            "bidirectional_model/*",
            "ar_distilled_action_model/*",
        ],
    )
    print(f"Downloaded to: {worldplay_path}")

    # Fix: Rename model.safetensors to diffusion_pytorch_model.safetensors
    # in ar_distilled_action_model (repo uses different filename)
    ar_distill_dir = os.path.join(worldplay_path, "ar_distilled_action_model")
    model_src = os.path.join(ar_distill_dir, "model.safetensors")
    model_dst = os.path.join(ar_distill_dir, "diffusion_pytorch_model.safetensors")

    if os.path.exists(model_src) and not os.path.exists(model_dst):
        real_src = os.path.realpath(model_src)
        shutil.copy2(real_src, model_dst)
        print(
            "Fixed: Renamed model.safetensors -> diffusion_pytorch_model.safetensors in ar_distilled_action_model"
        )

    return worldplay_path


def download_hunyuan_video(weights_root: str, hf_cache_dir: str):
    """Download HunyuanVideo-1.5 base models (vae, scheduler, transformer)."""
    print("\n" + "=" * 60)
    print("[2/6] Downloading tencent/HunyuanVideo-1.5 (vae, scheduler, transformer)...")
    print("=" * 60)

    hunyuan_path = _hf_snapshot_download(
        "tencent/HunyuanVideo-1.5",
        weights_root=weights_root,
        hf_cache_dir=hf_cache_dir,
        allow_patterns=["vae/*", "scheduler/*", "transformer/480p_i2v/*"],
    )
    print(f"Downloaded to: {hunyuan_path}")
    return hunyuan_path


def download_llm_text_encoder(hunyuan_path: str, weights_root: str, hf_cache_dir: str):
    """Download Qwen2.5-VL-7B-Instruct as the LLM text encoder."""
    print("\n" + "=" * 60)
    print("[3/6] Downloading LLM text encoder (Qwen2.5-VL-7B-Instruct)...")
    print("=" * 60)

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    llm_target = os.path.join(text_encoder_base, "llm")

    if (
        os.path.exists(llm_target)
        and os.path.isdir(llm_target)
        and len(os.listdir(llm_target)) > 5
    ):
        print(f"LLM text encoder already exists at: {llm_target}")
        return

    # Clean up old/broken downloads
    if os.path.islink(llm_target):
        os.unlink(llm_target)
    elif os.path.exists(llm_target):
        shutil.rmtree(llm_target)

    print("Downloading Qwen/Qwen2.5-VL-7B-Instruct (~15GB)...")
    # Download directly into the expected target folder.
    _hf_snapshot_download(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        weights_root=weights_root,
        hf_cache_dir=hf_cache_dir,
        allow_patterns=None,
    )
    # The repo itself is stored under weights_root/Qwen/Qwen2.5-VL-7B-Instruct.
    # We still need a copy under MODEL_PATH/text_encoder/llm to satisfy HY pipeline path checks.
    qwen_repo_dir = _repo_local_dir(weights_root, "Qwen/Qwen2.5-VL-7B-Instruct")
    os.makedirs(llm_target, exist_ok=True)
    for item in os.listdir(qwen_repo_dir):
        src = os.path.realpath(os.path.join(qwen_repo_dir, item))
        dst = os.path.join(llm_target, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"Copied to: {llm_target}")


def download_byt5_encoders(hunyuan_path: str, weights_root: str, hf_cache_dir: str):
    """Download ByT5 text encoders (byt5-small and Glyph-SDXL-v2)."""
    from modelscope import snapshot_download as ms_snapshot_download

    print("\n" + "=" * 60)
    print("[4/6] Downloading ByT5 text encoders...")
    print("=" * 60)

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    # 1. Download google/byt5-small
    byt5_target = os.path.join(text_encoder_base, "byt5-small")
    if (
        os.path.exists(byt5_target)
        and os.path.isdir(byt5_target)
        and len(os.listdir(byt5_target)) > 3
    ):
        print(f"byt5-small already exists at: {byt5_target}")
    else:
        if os.path.islink(byt5_target):
            os.unlink(byt5_target)
        elif os.path.exists(byt5_target):
            shutil.rmtree(byt5_target)

        print("Downloading google/byt5-small...")
        _hf_snapshot_download(
            "google/byt5-small",
            weights_root=weights_root,
            hf_cache_dir=hf_cache_dir,
            allow_patterns=None,
        )
        byt5_repo_dir = _repo_local_dir(weights_root, "google/byt5-small")
        os.makedirs(byt5_target, exist_ok=True)
        for item in os.listdir(byt5_repo_dir):
            src = os.path.realpath(os.path.join(byt5_repo_dir, item))
            dst = os.path.join(byt5_target, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {byt5_target}")

    # 2. Download Glyph-SDXL-v2 from ModelScope
    glyph_target = os.path.join(text_encoder_base, "Glyph-SDXL-v2")
    if os.path.exists(glyph_target) and os.path.exists(
        os.path.join(glyph_target, "checkpoints", "byt5_model.pt")
    ):
        print(f"Glyph-SDXL-v2 already exists at: {glyph_target}")
    else:
        if os.path.exists(glyph_target):
            shutil.rmtree(glyph_target)

        print("Downloading AI-ModelScope/Glyph-SDXL-v2 from ModelScope...")
        glyph_cache_dir = os.path.join(os.path.expanduser(weights_root), "_modelscope_cache")
        os.makedirs(glyph_cache_dir, exist_ok=True)
        glyph_cache = ms_snapshot_download("AI-ModelScope/Glyph-SDXL-v2", cache_dir=glyph_cache_dir)

        os.makedirs(glyph_target, exist_ok=True)
        for item in os.listdir(glyph_cache):
            src = os.path.join(glyph_cache, item)
            dst = os.path.join(glyph_target, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {glyph_target}")


def download_vision_encoder(hunyuan_path: str, hf_token: str, weights_root: str, hf_cache_dir: str):
    """Download SigLIP vision encoder from FLUX.1-Redux-dev."""
    print("\n" + "=" * 60)
    print("[5/6] Downloading Vision Encoder (SigLIP from FLUX.1-Redux-dev)...")
    print("=" * 60)

    if not hf_token:
        print("WARNING: No HF token provided!")
        print(
            "The vision encoder requires access to: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev"
        )
        print("Skipping vision encoder download.")
        print("\nYou can download it manually later.")
        return

    vision_encoder_base = os.path.join(hunyuan_path, "vision_encoder")
    os.makedirs(vision_encoder_base, exist_ok=True)

    siglip_target = os.path.join(vision_encoder_base, "siglip")

    if (
        os.path.exists(siglip_target)
        and os.path.isdir(siglip_target)
        and len(os.listdir(siglip_target)) > 3
    ):
        print(f"siglip already exists at: {siglip_target}")
        return

    # Clean up old/broken downloads
    if os.path.islink(siglip_target):
        os.unlink(siglip_target)
    elif os.path.exists(siglip_target):
        shutil.rmtree(siglip_target)

    print("Downloading black-forest-labs/FLUX.1-Redux-dev...")
    try:
        _hf_snapshot_download(
            "black-forest-labs/FLUX.1-Redux-dev",
            weights_root=weights_root,
            hf_cache_dir=hf_cache_dir,
            allow_patterns=None,
            token=hf_token,
        )
        flux_cache = _repo_local_dir(weights_root, "black-forest-labs/FLUX.1-Redux-dev")

        # Copy files (resolve symlinks)
        os.makedirs(siglip_target, exist_ok=True)
        for item in os.listdir(flux_cache):
            src = os.path.realpath(os.path.join(flux_cache, item))
            dst = os.path.join(siglip_target, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {siglip_target}")
    except Exception as e:
        print(f"ERROR: Failed to download vision encoder: {e}")
        print(
            "Make sure you have requested access to FLUX.1-Redux-dev and your token is valid."
        )


def print_paths(hunyuan_path: str, worldplay_path: str):
    """Print the model paths for run.sh configuration."""
    print("\n" + "=" * 60)
    print("[6/6] Verifying downloads...")
    print("=" * 60)
    if not os.path.isdir(hunyuan_path):
        raise RuntimeError(f"MODEL_PATH not found: {hunyuan_path}")
    if not os.path.isdir(worldplay_path):
        raise RuntimeError(f"HY-WorldPlay path not found: {worldplay_path}")

    print("\n" + "=" * 60)
    print("ALL DOWNLOADS COMPLETE!")
    print("=" * 60)
    print("\nAdd these paths to your run.sh:\n")
    print(f"MODEL_PATH={hunyuan_path}")
    print(
        f"AR_ACTION_MODEL_PATH={worldplay_path}/ar_model/diffusion_pytorch_model.safetensors"
    )
    print(
        f"BI_ACTION_MODEL_PATH={worldplay_path}/bidirectional_model/diffusion_pytorch_model.safetensors"
    )
    print(
        f"AR_DISTILL_ACTION_MODEL_PATH={worldplay_path}/ar_distilled_action_model/diffusion_pytorch_model.safetensors"
    )
    print("\nYou can now run: bash run.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Download all required models for HY-WorldPlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python download_models.py --hf_token hf_xxxxxxxxxxxxx

Note:
    The HuggingFace token is required for downloading the vision encoder
    from black-forest-labs/FLUX.1-Redux-dev. You need to:
    1. Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
    2. Wait for approval (usually instant)
    3. Create a token at: https://huggingface.co/settings/tokens (select "Read" permission)
        """,
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for downloading gated models (required for vision encoder)",
    )
    parser.add_argument(
        "--skip_vision_encoder",
        action="store_true",
        help="Skip downloading the vision encoder (if you don't have FLUX access yet)",
    )
    parser.add_argument(
        "--weights_root",
        type=str,
        default="~/alex/weights",
        help="Where to store downloaded weights, e.g. ~/alex/weights/tencent/HY-WorldPlay",
    )

    args = parser.parse_args()
    weights_root = os.path.expanduser(args.weights_root)
    _ensure_dir(weights_root)
    hf_cache_dir = _ensure_dir(os.path.join(weights_root, "_hf_cache"))

    print("=" * 60)
    print("HY-WorldPlay Model Download Script")
    print("=" * 60)

    # Check dependencies
    check_dependencies()

    # Download models
    worldplay_path = download_hy_worldplay(weights_root=weights_root, hf_cache_dir=hf_cache_dir)
    hunyuan_path = download_hunyuan_video(weights_root=weights_root, hf_cache_dir=hf_cache_dir)
    download_llm_text_encoder(hunyuan_path, weights_root=weights_root, hf_cache_dir=hf_cache_dir)
    download_byt5_encoders(hunyuan_path, weights_root=weights_root, hf_cache_dir=hf_cache_dir)

    if not args.skip_vision_encoder:
        download_vision_encoder(
            hunyuan_path,
            args.hf_token,
            weights_root=weights_root,
            hf_cache_dir=hf_cache_dir,
        )
    else:
        print("\n[5/6] Skipping vision encoder download (--skip_vision_encoder flag)")

    # Print final paths
    print_paths(hunyuan_path=hunyuan_path, worldplay_path=worldplay_path)


if __name__ == "__main__":
    main()
