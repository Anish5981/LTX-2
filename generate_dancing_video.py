import argparse
import logging
import torch
from pathlib import Path

# Assuming ltx-pipelines is installed in editable mode or in the python path
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.quantization import QuantizationPolicy

def generate_dance(
    checkpoint_path,
    upsampler_path,
    gemma_root,
    ic_lora_path,
    image_path,
    video_motion_path,
    prompt,
    output_path="dance_output.mp4",
    num_frames=377,  # 15s @ 25fps (8*47 + 1)
    height=1024,
    width=1536,
    seed=42,
    use_fp8=True
):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        logging.warning("CUDA not available. Generation will be extremely slow or fail.")

    # 1. Setup LoRA
    loras = [
        LoraPathStrengthAndSDOps(
            ic_lora_path,
            1.0,
            LTXV_LORA_COMFY_RENAMING_MAP
        )
    ]

    # 2. Initialize Pipeline
    quantization = QuantizationPolicy.fp8_cast() if use_fp8 else None
    
    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=checkpoint_path,
        spatial_upsampler_path=upsampler_path,
        gemma_root=gemma_root,
        loras=loras,
        device=device,
        quantization=quantization,
    )

    # 3. Setup Inputs
    # Condition on images at frame 0
    images = [
        ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)
    ]
    
    # Motion video conditioning
    video_conditioning = [
        (video_motion_path, 1.0)
    ]

    # 4. Generate
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    
    logging.info(f"Starting generation for {num_frames} frames ({num_frames/25:.1f}s)...")
    
    video, audio = pipeline(
        prompt=prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=25.0,
        images=images,
        video_conditioning=video_conditioning,
        tiling_config=tiling_config,
    )

    # 5. Save Output
    logging.info(f"Saving video to {output_path}...")
    encode_video(
        video=video,
        fps=25.0,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )
    logging.info("Generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 15-second dancing girl video using LTX-2.")
    parser.add_argument("--checkpoint", required=True, help="Path to ltx-2.3-22b-distilled.safetensors")
    parser.add_argument("--upsampler", required=True, help="Path to ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
    parser.add_argument("--gemma", required=True, help="Path to Gemma 3 root directory")
    parser.add_argument("--lora", required=True, help="Path to IC-LoRA Union or Pose Control safetensors")
    parser.add_argument("--image", required=True, help="Path to the source image of the girl")
    parser.add_argument("--video", required=True, help="Path to the motion reference video")
    parser.add_argument("--prompt", default="A girl dancing gracefully, high quality, cinematic lighting", help="Text prompt")
    parser.add_argument("--output", default="dance_output.mp4", help="Output video path")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 quantization")
    
    args = parser.parse_args()
    
    generate_dance(
        checkpoint_path=args.checkpoint,
        upsampler_path=args.upsampler,
        gemma_root=args.gemma,
        ic_lora_path=args.lora,
        image_path=args.image,
        video_motion_path=args.video,
        prompt=args.prompt,
        output_path=args.output,
        use_fp8=not args.no_fp8
    )
