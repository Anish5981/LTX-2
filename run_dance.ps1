param(
    [string]$checkpoint = "path/to/ltx-2.3-22b-distilled.safetensors",
    [string]$upsampler = "path/to/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    [string]$gemma = "path/to/gemma",
    [string]$lora = "path/to/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    [string]$image = "path/to/girl.jpg",
    [string]$video = "path/to/dance_motion.mp4",
    [string]$prompt = "A girl dancing gracefully, high quality, cinematic lighting",
    [string]$output = "dance_output.mp4"
)

# Set environment variables for memory optimization
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Write-Host "--- LTX-2 Dancing Girl Generation ---" -ForegroundColor Cyan
Write-Host "Using Checkpoint: $checkpoint"
Write-Host "Using Upsampler: $upsampler"
Write-Host "Using LoRA: $lora"
Write-Host "Source Image: $image"
Write-Host "Motion Video: $video"
Write-Host "Prompt: $prompt"
Write-Host "-------------------------------------"

python generate_dancing_video.py `
    --checkpoint "$checkpoint" `
    --upsampler "$upsampler" `
    --gemma "$gemma" `
    --lora "$lora" `
    --image "$image" `
    --video "$video" `
    --prompt "$prompt" `
    --output "$output"

Write-Host "Done! If successful, check $output" -ForegroundColor Green
