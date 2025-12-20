# SPDX-License-Identifier: Apache-2.0
import json
import os

import torch
import pytest

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)

device_name = torch.cuda.get_device_name()
device_reference_folder_suffix = '_reference_videos'

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
else:
    # Default to L40S reference videos for other devices (e.g., H100)
    device_reference_folder = "L40S" + device_reference_folder_suffix

# Base parameters from the shell script
HUNYUAN_PARAMS = {
    "num_gpus": 2,
    "model_path": "FastVideo/FastHunyuan-diffusers",
    "height": 720,
    "width": 1280,
    "num_frames": 45,
    "num_inference_steps": 6,
    "guidance_scale": 1,
    "embedded_cfg_scale": 6,
    "flow_shift": 17,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 1,
    "vae_sp": True,
    "fps": 24,
}

WAN_T2V_PARAMS = {
    "num_gpus": 2,
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 20,
    "guidance_scale": 3,
    "embedded_cfg_scale": 6,
    "flow_shift": 7.0,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 1,
    "vae_sp": True,
    "fps": 24,
    "neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    "text-encoder-precision": ("fp32",)
}

WAN_I2V_PARAMS = {
    "num_gpus": 2,
    "model_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 6,
    "guidance_scale": 5.0,
    "embedded_cfg_scale": 6,
    "flow_shift": 7.0,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 1,
    "vae_sp": True,
    "fps": 24,
    "neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    "text-encoder-precision": ("fp32",)
}

# LongCat T2V 50-step original parameters
LONGCAT_T2V_ORIGINAL_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/LongCat-Video-T2V-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 93,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "seed": 42,
    "sp_size": 1,
    "tp_size": 1,
    "fps": 15,
    "neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
}

# LongCat Distilled parameters (16 steps, guidance_scale=1.0)
LONGCAT_T2V_DISTILLED_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/LongCat-Video-T2V-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 93,
    "num_inference_steps": 16,
    "guidance_scale": 1.0,
    "seed": 42,
    "sp_size": 1,
    "tp_size": 1,
    "fps": 15,
    "neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    "lora_nickname": "distilled",
    "lora_path": "FastVideo/LongCat-Video-T2V-Distilled-LoRA",
}

# LongCat Refine parameters (50 steps, 720p output)
LONGCAT_T2V_REFINE_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/LongCat-Video-T2V-Diffusers",
    "height": 720,
    "width": 1280,
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
    "seed": 42,
    "sp_size": 1,
    "tp_size": 1,
    "fps": 30,
    "lora_nickname": "refinement",
    "lora_path": "FastVideo/LongCat-Video-T2V-Refinement-LoRA",
    "enable_bsa": True,
    "bsa_sparsity": 0.875,
    "t_thresh": 0.5,
}

MODEL_TO_PARAMS = {
    "FastHunyuan-diffusers": HUNYUAN_PARAMS,
    "Wan2.1-T2V-1.3B-Diffusers": WAN_T2V_PARAMS,
}

I2V_MODEL_TO_PARAMS = {
    "Wan2.1-I2V-14B-480P-Diffusers": WAN_I2V_PARAMS,
}

LONGCAT_T2V_MODEL_TO_PARAMS = {
    "LongCat-Video-T2V-Original": LONGCAT_T2V_ORIGINAL_PARAMS,
}

LONGCAT_T2V_DISTILLED_MODEL_TO_PARAMS = {
    "LongCat-Video-T2V-Distilled": LONGCAT_T2V_DISTILLED_PARAMS,
}

TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
    # "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature."
]

I2V_TEST_PROMPTS = [
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
]

I2V_IMAGE_PATHS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
]

LONGCAT_T2V_TEST_PROMPTS = [
    "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene.",
]



@pytest.mark.parametrize("prompt", I2V_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(I2V_MODEL_TO_PARAMS.keys()))
def test_i2v_inference_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Test that runs inference with different parameters and compares the output
    to reference videos using SSIM.
    """
    assert len(I2V_TEST_PROMPTS) == len(I2V_IMAGE_PATHS), "Expect number of prompts equal to number of images"
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_output_dir = os.path.join(script_dir, 'generated_videos', model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip()}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = I2V_MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]
    image_path = I2V_IMAGE_PATHS[I2V_TEST_PROMPTS.index(prompt)]

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "flow_shift": BASE_PARAMS["flow_shift"],
        "sp_size": BASE_PARAMS["sp_size"],
        "tp_size": BASE_PARAMS["tp_size"],
    }
    if BASE_PARAMS.get("vae_sp"):
        init_kwargs["vae_sp"] = True
        init_kwargs["vae_tiling"] = True
    if "text-encoder-precision" in BASE_PARAMS:
        init_kwargs["text_encoder_precisions"] = BASE_PARAMS["text-encoder-precision"]

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "image_path": image_path,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "embedded_cfg_scale": BASE_PARAMS["embedded_cfg_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
    }
    if "neg_prompt" in BASE_PARAMS:
        generation_kwargs["neg_prompt"] = BASE_PARAMS["neg_prompt"]

    generator = VideoGenerator.from_pretrained(model_path=BASE_PARAMS["model_path"], **init_kwargs)
    generator.generate_video(prompt, **generation_kwargs)
    
    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

    assert os.path.exists(
        output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = os.path.join(script_dir, device_reference_folder, model_id, ATTENTION_BACKEND)

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}")

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(f"Reference video not found for prompt: {prompt} with backend: {ATTENTION_BACKEND}")
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                 generated_video_path,
                                                 use_ms_ssim=True)

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(output_dir, ssim_values, reference_video_path,
                                 generated_video_path, num_inference_steps,
                                 prompt)

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.97
    assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for {model_id} with backend {ATTENTION_BACKEND}"

@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN", "TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_inference_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Test that runs inference with different parameters and compares the output
    to reference videos using SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_output_dir = os.path.join(script_dir, 'generated_videos', model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip()}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "flow_shift": BASE_PARAMS["flow_shift"],
        "sp_size": BASE_PARAMS["sp_size"],
        "tp_size": BASE_PARAMS["tp_size"],
        "dit_cpu_offload": True,
    }
    if BASE_PARAMS.get("vae_sp"):
        init_kwargs["vae_sp"] = True
        init_kwargs["vae_tiling"] = True
    if "text-encoder-precision" in BASE_PARAMS:
        init_kwargs["text_encoder_precisions"] = BASE_PARAMS["text-encoder-precision"]

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "embedded_cfg_scale": BASE_PARAMS["embedded_cfg_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
    }
    if "neg_prompt" in BASE_PARAMS:
        generation_kwargs["neg_prompt"] = BASE_PARAMS["neg_prompt"]

    generator = VideoGenerator.from_pretrained(model_path=BASE_PARAMS["model_path"], **init_kwargs)
    generator.generate_video(prompt, **generation_kwargs)

    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

    assert os.path.exists(
        output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = os.path.join(script_dir, device_reference_folder, model_id, ATTENTION_BACKEND)

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}")

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(f"Reference video not found for prompt: {prompt} with backend: {ATTENTION_BACKEND}")
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                 generated_video_path,
                                                 use_ms_ssim=True)

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(output_dir, ssim_values, reference_video_path,
                                 generated_video_path, num_inference_steps,
                                 prompt)

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.93
    assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for {model_id} with backend {ATTENTION_BACKEND}"


@pytest.mark.parametrize("prompt", LONGCAT_T2V_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(LONGCAT_T2V_MODEL_TO_PARAMS.keys()))
def test_longcat_t2v_original_inference_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Test LongCat T2V 50-step original inference.
    Compares output to reference videos using SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_output_dir = os.path.join(script_dir, 'generated_videos', model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip()}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = LONGCAT_T2V_MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "sp_size": BASE_PARAMS["sp_size"],
        "tp_size": BASE_PARAMS["tp_size"],
        "dit_cpu_offload": True,
        "vae_cpu_offload": True,
        "text_encoder_cpu_offload": True,
    }

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
        "neg_prompt": BASE_PARAMS["neg_prompt"],
    }

    generator = VideoGenerator.from_pretrained(model_path=BASE_PARAMS["model_path"], **init_kwargs)
    generator.generate_video(prompt, **generation_kwargs)

    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

    generated_video_path = os.path.join(output_dir, output_video_name)
    assert os.path.exists(generated_video_path), f"Output video was not generated at {generated_video_path}"

    reference_folder = os.path.join(script_dir, device_reference_folder, model_id, ATTENTION_BACKEND)

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(f"Reference video folder does not exist: {reference_folder}")

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(f"Reference video not found for prompt: {prompt} with backend: {ATTENTION_BACKEND}")
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)

    logger.info(f"Computing SSIM between {reference_video_path} and {generated_video_path}")
    ssim_values = compute_video_ssim_torchvision(reference_video_path, generated_video_path, use_ms_ssim=True)

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(
        output_dir, ssim_values, reference_video_path,
        generated_video_path, num_inference_steps, prompt
    )

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.93
    assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for {model_id} with backend {ATTENTION_BACKEND}"


@pytest.mark.parametrize("prompt", LONGCAT_T2V_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(LONGCAT_T2V_DISTILLED_MODEL_TO_PARAMS.keys()))
def test_longcat_t2v_distilled_refine_inference_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Test LongCat T2V distilled (16-step) + refinement inference.
    
    This test:
    1. Generates a 480p video using distilled LoRA (16 steps)
    2. Refines the 480p video to 720p using refinement LoRA (50 steps)
    3. Compares the final output to reference videos using SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_PARAMS = LONGCAT_T2V_DISTILLED_MODEL_TO_PARAMS[model_id]
    REFINE_PARAMS = LONGCAT_T2V_REFINE_PARAMS

    # Step 1: Generate distilled 480p video
    distilled_output_dir = os.path.join(script_dir, 'generated_videos', model_id, ATTENTION_BACKEND)
    os.makedirs(distilled_output_dir, exist_ok=True)

    distilled_video_name = f"{prompt[:100].strip()}.mp4"
    distilled_video_path = os.path.join(distilled_output_dir, distilled_video_name)

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "sp_size": BASE_PARAMS["sp_size"],
        "tp_size": BASE_PARAMS["tp_size"],
        "dit_cpu_offload": True,
        "vae_cpu_offload": True,
        "text_encoder_cpu_offload": True,
    }

    generation_kwargs = {
        "num_inference_steps": BASE_PARAMS["num_inference_steps"],
        "output_path": distilled_output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
        "neg_prompt": BASE_PARAMS["neg_prompt"],
    }

    generator = VideoGenerator.from_pretrained(model_path=BASE_PARAMS["model_path"], **init_kwargs)
    
    # Load distilled LoRA
    generator.set_lora_adapter(
        lora_nickname=BASE_PARAMS["lora_nickname"],
        lora_path=BASE_PARAMS["lora_nickname"]  # Uses model's lora directory
    )
    
    generator.generate_video(prompt, **generation_kwargs)

    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

    assert os.path.exists(distilled_video_path), f"Distilled video was not generated at {distilled_video_path}"

    # Step 2: Refine to 720p
    refined_output_dir = os.path.join(script_dir, 'generated_videos', f"{model_id}-Refined", ATTENTION_BACKEND)
    os.makedirs(refined_output_dir, exist_ok=True)

    refined_video_name = f"{prompt[:100].strip()}.mp4"
    refined_video_path = os.path.join(refined_output_dir, refined_video_name)

    refine_init_kwargs = {
        "num_gpus": REFINE_PARAMS["num_gpus"],
        "sp_size": REFINE_PARAMS["sp_size"],
        "tp_size": REFINE_PARAMS["tp_size"],
        "dit_cpu_offload": True,
        "vae_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "enable_bsa": REFINE_PARAMS["enable_bsa"],
    }

    refine_generation_kwargs = {
        "num_inference_steps": REFINE_PARAMS["num_inference_steps"],
        "output_path": refined_output_dir,
        "height": REFINE_PARAMS["height"],
        "width": REFINE_PARAMS["width"],
        "guidance_scale": REFINE_PARAMS["guidance_scale"],
        "seed": REFINE_PARAMS["seed"],
        "fps": REFINE_PARAMS["fps"],
        "refine_from": distilled_video_path,
        "t_thresh": REFINE_PARAMS["t_thresh"],
    }

    refine_generator = VideoGenerator.from_pretrained(model_path=REFINE_PARAMS["model_path"], **refine_init_kwargs)
    
    # Load refinement LoRA
    refine_generator.set_lora_adapter(
        lora_nickname=REFINE_PARAMS["lora_nickname"],
        lora_path=REFINE_PARAMS["lora_nickname"]  # Uses model's lora directory
    )
    
    refine_generator.generate_video(prompt, **refine_generation_kwargs)

    if isinstance(refine_generator.executor, MultiprocExecutor):
        refine_generator.executor.shutdown()

    assert os.path.exists(refined_video_path), f"Refined video was not generated at {refined_video_path}"

    # Find reference video for refined output
    reference_folder = os.path.join(script_dir, device_reference_folder, f"{model_id}-Refined", ATTENTION_BACKEND)

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(f"Reference video folder does not exist: {reference_folder}")

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(f"Reference video not found for prompt: {prompt} with backend: {ATTENTION_BACKEND}")
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)

    logger.info(f"Computing SSIM between {reference_video_path} and {refined_video_path}")
    ssim_values = compute_video_ssim_torchvision(reference_video_path, refined_video_path, use_ms_ssim=True)

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {refined_output_dir}")

    success = write_ssim_results(
        refined_output_dir, ssim_values, reference_video_path,
        refined_video_path, REFINE_PARAMS["num_inference_steps"], prompt
    )

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for {model_id} Distilled+Refine with backend {ATTENTION_BACKEND}"
