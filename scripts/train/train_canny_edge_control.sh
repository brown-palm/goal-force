# for 81 frames
#   max = 6  when using accelerate_config_4_gpu_zero_stage_2.yaml
#   max = 10 when using accelerate_config_4_gpu_zero_stage_2_offload_optimizer.yaml
CONTROLNET_NUM_LAYERS=10

CONTROL_SIGNAL_TYPE="canny_edge"
DATASET_BASE_PATH="datasets/OpenVid-1M/OpenVid-1M-train/video"
DATASET_METADATA_PATH="datasets/OpenVid-1M/OpenVid-1M-train/OpenVid-1M.csv"

accelerate launch \
  --config_file scripts/accelerate/accelerate_config_4_gpu_zero_stage_2_offload_optimizer.yaml \
  scripts/train/train.py \
  --dataset_base_path ${DATASET_BASE_PATH} \
  --dataset_metadata_path ${DATASET_METADATA_PATH} \
  --control_signal_type ${CONTROL_SIGNAL_TYPE} \
  --controlnet_num_layers ${CONTROLNET_NUM_LAYERS} \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --save_steps 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models "controlnet" \
  --output_path "outputs/${CONTROL_SIGNAL_TYPE}" \
  --extra_inputs "input_image" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0 \
  --max_grad_norm 1 \
  --wandb_logging