from fastvideo import VideoGenerator

# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_wan_action_i2v"

def main():
    # Load WanActionTransformer3DModel with discrete action parameters
    # This is equivalent to the HY-WorldPlay approach:
    #   transformer = WanTransformer3DModel.from_pretrained(...)
    #   transformer.add_discrete_action_parameters()
    #
    # We achieve this by:
    #   1. override_transformer_cls_name="WanActionTransformer3DModel" - loads from wanaction.py
    #   2. add_action_parameters=True - calls add_discrete_action_parameters() after loading
    
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        # Override the transformer class to use WanActionTransformer3DModel from wanaction.py
        override_transformer_cls_name="WanActionTransformer3DModel",
        # Call add_discrete_action_parameters() after loading the transformer
        add_action_parameters=True,
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        # image_encoder_cpu_offload=False,
    )

    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    image_path = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"

    video = generator.generate_video(
        prompt,
        image_path=image_path,
        output_path=OUTPUT_PATH,
        save_video=True,
        height=832,
        width=480,
        num_frames=81,
    )


if __name__ == "__main__":
    main()
