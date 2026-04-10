from huggingface_hub import hf_hub_download
import os

def download_ltx_models(target_dir="models"):
    os.makedirs(target_dir, exist_ok=True)
    
    files_to_download = [
        # Model & Upscaler
        ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled.safetensors"),
        ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
        # IC-LoRA
        ("Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control", "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
    ]
    
    print(f"Starting download to {target_dir}...")
    print("WARNING: This will download ~25GB of data.")
    
    for repo_id, filename in files_to_download:
        print(f"Downloading {filename} from {repo_id}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to: {path}")

    # Gemma 3 is a whole directory, better to use snapshot_download or just direct instructions
    print("\n--- Manual Step Required ---")
    print("Gemma 3 text encoder must be downloaded separately due to its size and many files.")
    print("Run this command in your terminal:")
    print("huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir models/gemma-3")

if __name__ == "__main__":
    try:
        import huggingface_hub
    except ImportError:
        print("Please install huggingface_hub first: pip install huggingface_hub")
        exit(1)
        
    download_ltx_models()
