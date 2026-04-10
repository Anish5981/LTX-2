import torch
import logging
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.quantization import QuantizationPolicy

# Minimalist Test Run for 4GB VRAM
# This attempts to generate 1 second of video at low resolution using layer streaming.

def run_minimal_test(
    checkpoint_path="models/ltx-2.3-22b-distilled.safetensors",
    upsampler_path="models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    gemma_root="models/gemma-3",
    ic_lora_path="models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    image_path="test_image.jpg",
    video_motion_path="test_motion.mp4",
):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda")
    
    # 1. Setup LoRA
    loras = [
        LoraPathStrengthAndSDOps(ic_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)
    ]

    # 2. Initialize Pipeline with ALL the optimizations
    # Layer streaming prefetch count set to 1 (uses minimal VRAM by swapping layers)
    # FP8 quantization enabled
    quantization = QuantizationPolicy.fp8_cast()
    
    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=checkpoint_path,
        spatial_upsampler_path=upsampler_path,
        gemma_root=gemma_root,
        loras=loras,
        device=device,
        quantization=quantization,
    )

    # 3. Setup Inputs (Tiny 1 second test)
    num_frames = 25  # 1 second @ 25 fps
    height = 256     # Low res to avoid VAE OOM
    width = 384
    images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]
    video_conditioning = [(video_motion_path, 1.0)]

    # 4. Generate
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    
    logging.info(f"TEST RUN: Attempting 1 second (25 frames) at {height}x{width}...")
    logging.info("This will likely be slow due to layer streaming on 4GB VRAM.")
    
    video, audio = pipeline(
        prompt="A girl dancing, minimal test run",
        seed=42,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=25.0,
        images=images,
        video_conditioning=video_conditioning,
        tiling_config=tiling_config,
        streaming_prefetch_count=1, # CRITICAL: Enables layer streaming
    )

    # 5. Save Output
    encode_video(
        video=video,
        fps=25.0,
        audio=audio,
        output_path="test_minimal_output.mp4",
        video_chunks_number=video_chunks_number,
    )
    logging.info("Minimal test completed successfully!")

if __name__ == "__main__":
    # Note: These paths assume you have downloaded them into the 'models' folder.
    run_minimal_test()
