import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from diffsynth import save_video
from PIL import Image

import sys
for path in sys.path:
    path_to_this_file = "scripts/inference"
    if path.endswith(path_to_this_file):
        sys.path.append(path.replace(path_to_this_file, ""))
        break

from src.goal_force.wan_video_new import WanVideoPipeline, ModelConfig
from src.goal_force.unified_dataset import (
    ControlSignalDataset_CannyEdge, 
    ControlSignalDataset_Balls,
)
import pandas as pd
import numpy as np
from src.goal_force.utils import safe_collate, add_aesthetic_point_force_prompt_to_video
import argparse
import json

from scripts.inference.utils import split_list_across_devices_contiguous

# CHANGE WITH CAUTION: These must agree with the model you're loading
CONTROLNET_NUM_LAYERS = 10
NUM_FRAMES = 49

SKIP_MODEL_LOADING_FOR_DEBUGGING_DATA = False
TORCH_DTYPE = torch.bfloat16
OFFLOAD_DEVICE="cpu"

DATASET_CONSTRUCTOR = {
    "canny_edge": ControlSignalDataset_CannyEdge,
}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0,
                        help='Device ID (0-5 for 6 MIG instances)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of devices/processes (for CSV partitioning)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for video generation (default: 0)')
    parser.add_argument('--control_signal_type', type=str, 
                        help='Type of control signal to use')
    parser.add_argument('--model_ckpt_path', type=str, required=True,
                        help='Path to the model checkpoint file')
    parser.add_argument('--example_paths', type=str, nargs='+', required=True,
                        help='Path(s) to example file(s). For goal_force: CSV file(s). For canny_edge: video file(s).')
                        
    return parser.parse_args()



def main(args):
    DatasetConstructor = DATASET_CONSTRUCTOR[args.control_signal_type]

    # If CUDA_VISIBLE_DEVICES is set, the isolated device appears as cuda:0
    # Otherwise, use the device_id directly
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device = "cuda:0"
    else:
        device = f"cuda:{args.device_id}"

    # Diagnostic output
    print(f"[Device {args.device_id}] Initialized inference:")
    print(f"[Device {args.device_id}]   - World size: {args.world_size}")
    print(f"[Device {args.device_id}]   - Device: {args.device_id}")
    print(f"[Device {args.device_id}]   - Seed: {args.seed}")

    ckpt_dir_controlnet = os.path.dirname(args.model_ckpt_path)
    step_num = os.path.basename(args.model_ckpt_path).split(".safetensors")[0].split("-")[-1]
    step_dir = os.path.join(ckpt_dir_controlnet, f"step-{step_num}-videos")
    os.makedirs(step_dir, exist_ok=True)

    if not SKIP_MODEL_LOADING_FOR_DEBUGGING_DATA:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=TORCH_DTYPE,
            device=device,
            tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*", path="./models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device=OFFLOAD_DEVICE, path=['./models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors', 
                                                                                                                                                                        './models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors', 
                                                                                                                                                                        './models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors', 
                                                                                                                                                                        './models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors', 
                                                                                                                                                                        './models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors', 
                                                                                                                                                                        './models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors']),
                ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device=OFFLOAD_DEVICE, path=['./models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors', 
                                                                                                                                                                            './models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors', 
                                                                                                                                                                            './models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors', 
                                                                                                                                                                            './models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors', 
                                                                                                                                                                            './models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors', 
                                                                                                                                                                            './models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors']),
                ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device=OFFLOAD_DEVICE, path="./models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device=OFFLOAD_DEVICE, path="./models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
            ],
            controlnet=True,
            controlnet_num_layers=CONTROLNET_NUM_LAYERS,
        )

        # load checkpoint for each controlnet. not loading weights results in using zero-initialized zero-convs, 
        # which is equivalent to wan2.2 inference (albiet with a larger memory footprint)

        pipe.load_controlnet_weights(
            pipe.controlnet, args.model_ckpt_path, torch_dtype=TORCH_DTYPE)

        pipe.enable_vram_management()

    # Split examples across devices
    device_examples = split_list_across_devices_contiguous(args.example_paths, args.world_size, args.device_id)
    print(f"\n\n[Device {args.device_id}, seed {args.seed}] Processing {len(device_examples)} out of {len(args.example_paths)} examples. Processing: ", device_examples)

    assert args.control_signal_type == "canny_edge"

    for csv in device_examples:
        print("Doing csv: ", csv)
        df = pd.read_csv(csv)
    
        base_path = os.path.dirname(csv)
        image_folder = os.path.join(base_path, "images")
        canny_video_folder = os.path.join(base_path, "canny-videos")
        
        for _, row in df.iterrows():
            image_name = row["image"]
            control_video_name = row["control_video"]
            prompt = row["caption"]
            
            # Load input image
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            input_image = Image.open(image_path).convert("RGB")
            
            # Load control signal video using dataset's video operator
            video_operator = DatasetConstructor.default_video_operator(
                base_path=canny_video_folder,
                max_pixels=921600,
                height=480,
                width=832,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=NUM_FRAMES,
                time_division_factor=4,
                time_division_remainder=1,
            )
            processed_control_video = video_operator(control_video_name)
            
            control_video_np = np.array(processed_control_video)  # (num_frames, 480, 832, 3), uint8 [0, 255]
            control_signal_video = torch.from_numpy(control_video_np).to(torch.float32) / 127.5 - 1.0
            control_signal_video = control_signal_video.to(torch.bfloat16)
            
            # Define output filenames
            video_path_root = control_video_name.split("_canny.mp4")[0].split(".mp4")[0]
            output_fname_video = os.path.join(step_dir, f"{video_path_root}-canny-output.mp4")
            fname_control_video     = os.path.join(step_dir, f"{video_path_root}-canny-control-signal.mp4")
            fname_image_condition   = os.path.join(step_dir, f"{video_path_root}-image-condition.png")

            # save first frame
            input_image.save(fname_image_condition)
            input_image = input_image.convert("RGB")

            # save control signal video
            control_signal_video_ = (control_signal_video.to(float).numpy() * 255).astype(np.uint8) #[0, 1] --> {0, 1, ..., 255}
            save_video(control_signal_video_, fname_control_video, fps=15, quality=5)
            
            if not SKIP_MODEL_LOADING_FOR_DEBUGGING_DATA:
                # Run inference
                video = pipe(
                    prompt=prompt,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    input_image=input_image,
                    num_frames=NUM_FRAMES,
                    seed=args.seed,
                    tiled=True,
                    controlnet=True,
                    control_signal_video=control_signal_video.to(device)
                )
                save_video(video, output_fname_video, fps=15, quality=5)


if __name__ == "__main__":
    args = parse_args()
    main(args)