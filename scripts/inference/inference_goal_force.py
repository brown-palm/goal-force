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
    ControlSignalDataset_Balls,
)
import numpy as np
from src.goal_force.utils import safe_collate, add_aesthetic_point_force_prompt_to_video
import argparse
import json

from scripts.inference.utils import split_list_across_devices_contiguous

# CHANGE WITH CAUTION: These must agree with the model you're loading
CONTROLNET_NUM_LAYERS = 10
NUM_FRAMES = 81

SKIP_MODEL_LOADING_FOR_DEBUGGING_DATA = False
TORCH_DTYPE = torch.bfloat16
OFFLOAD_DEVICE="cpu"

DATASET_CONSTRUCTOR = {
    "goal_force": ControlSignalDataset_Balls,
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
                        choices=['goal_force', 'canny_edge'],
                        default='goal_force',
                        help='Type of control signal to use (default: goal_force)')
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

    assert args.control_signal_type == "goal_force"

    for csv in device_examples:
        print("Doing csv: ", csv)

        base_path = os.path.dirname(csv)
        metadata_path = csv

        # create validation dataset and dataloader
        dataset = DatasetConstructor(
            base_path=base_path,
            metadata_path=metadata_path,
            is_validation_dataset=True,
            num_frames=NUM_FRAMES,
            height=480,
            width=832,
        )

        # NOTE: Hardcoding to overwrite min mass and max mass, calibrating from what was used in training
        print("WARNING: Hardcoding min/max mass and force values to match training dataset.")
        dataset.min_mass = 1.0
        dataset.max_mass = 4.0
        dataset.min_force = 30.0
        dataset.max_force = 400.0

        # we assume goal force is normalized to be the same...
        dataset.min_indirect_force = dataset.min_force
        dataset.max_indirect_force = dataset.max_force

        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=safe_collate, num_workers=0)

        for data in dataloader:
            # extract attributes needed for inference
            prompt                  = data["prompt"]
            input_image             = data["video"]
            control_signal_video    = data["control_video"]
            assert(len(input_image) == 1) # e.g. [<PIL.Image.Image image mode=RGB size=832x480 at 0x7F5379ACC1D0>] 

            # create file output names
            file_id                 = data["file_id"]
            coords_projectile       = data["coords"]["projectile"]
            coords_target           = data["coords"]["target"]

            prj_force               = data["force"]
            prj_angle               = data["angle"]
            prj_x_pos               = data["x_pos"]
            prj_y_pos               = data["y_pos"]
            prj_mass                = data["masses"]["projectile"]

            tgt_indirect_force      = data["target_indirect_force"]
            tgt_indirect_angle      = data["target_indirect_angle"]
            tgt_x_pos               = data["target_x_pos"]
            tgt_y_pos               = data["target_y_pos"]
            tgt_mass                = data["masses"]["target"]
            
            # create fname string, used for fname roots
            fname_str    = f"step-{step_num}_{file_id}"
            fname_str   += f"__prj_coords_{prj_x_pos:.2f}_{prj_y_pos:.2f}"
            fname_str   += f"__tgt_coords_{tgt_x_pos:.2f}_{tgt_y_pos:.2f}"
            fname_str   += f"__prj_mass_{prj_mass:.1f}"
            fname_str   += f"__tgt_mass_{tgt_mass:.1f}"
            fname_str   += f"__prj_force_{prj_force:.1f}__prj_angle_{prj_angle:.1f}"
            fname_str   += f"__tgt_indirect_force_{tgt_indirect_force:.1f}__tgt_indirect_angle_{tgt_indirect_angle:.1f}"
            fname_str   += f"__seed_{args.seed}"

            print(f"\ncurrently working on fname_str:\n{fname_str}\n")

            # create fnames for everything we want to save
            fname_control_video             = os.path.join(step_dir, f"{fname_str}-control-signal.mp4")
            fname_image_condition           = os.path.join(step_dir, f"{fname_str}-image_condition.png")
            fname_output_video              = os.path.join(step_dir, f"{fname_str}.mp4")
            fname_output_video_with_prompt  = os.path.join(step_dir, f"{fname_str}-with-prompt.mp4")
            fname_text                      = os.path.join(step_dir, f"{fname_str}-text.json")

            # save first frame
            input_image[0].save(fname_image_condition)
            input_image = input_image[0].convert("RGB")

            # save control signal video
            control_signal_video_ = (control_signal_video.to(float).numpy() * 255).astype(np.uint8) #[0, 1] --> {0, 1, ..., 255}
            save_video(control_signal_video_, fname_control_video, fps=15, quality=5)

            # save the text prompt as well
            text_prompt = {"text_prompt" : prompt}
            with open(fname_text, 'w') as f:
                json.dump(text_prompt, f, indent=4)

            if not SKIP_MODEL_LOADING_FOR_DEBUGGING_DATA:
                # generate the video...
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
                # ... and then save it
                save_video(video, fname_output_video, fps=15, quality=5)

                # save video with pretty force prompt
                video_array = np.asarray(torch.stack([dataset.to_tensor_transform(image) for image in video]))
                video_array = np.moveaxis(video_array, 1, -1)

                # if projectile force is specified, we make a red arrow
                if prj_force >  -1:
                    force_normalized = (prj_force - dataset.min_force) / (dataset.max_force - dataset.min_force)
                    color_red = (255, 0, 0)
                    video_with_prompt = add_aesthetic_point_force_prompt_to_video(
                        video_array, force_normalized, prj_angle, prj_x_pos, 1-prj_y_pos, circle_radius=20, num_frames_with_signal=16, color=color_red)

                # if target force is specified, we make a green arrow (green for goal!)
                if tgt_indirect_force >  -1:
                    color_green = (0, 255, 0)
                    force_normalized = (tgt_indirect_force - dataset.min_force) / (dataset.max_force - dataset.min_force)
                    video_with_prompt = add_aesthetic_point_force_prompt_to_video(
                        video_array, force_normalized, tgt_indirect_angle, tgt_x_pos, 1-tgt_y_pos, circle_radius=20, num_frames_with_signal=16, color=color_green)

                video_with_prompt = [dataset.to_pil_transform(frame) for frame in video_with_prompt]
                save_video(video_with_prompt, fname_output_video_with_prompt, fps=15, quality=5)


if __name__ == "__main__":
    args = parse_args()
    main(args)