# for 81 frames
#   max = 6  when using accelerate_config_4_gpu_zero_stage_2.yaml
#   max = 10 when using accelerate_config_4_gpu_zero_stage_2_offload_optimizer.yaml
CONTROLNET_NUM_LAYERS=10

CKPT_PATH="path/to/checkpoint.safetensors"

CONTROL_SIGNAL_TYPE="direct_force_and_goal_force_and_mass"

DATASET_BASE_PATH_BALLS="datasets/train/balls_6k"
DATASET_METADATA_PATH_BALLS="datasets/train/balls_6k.csv"

DATASET_BASE_PATH_DOMINOS="datasets/train/dominos_3k"
DATASET_METADATA_PATH_DOMINOS="datasets/train/dominos_3k.csv"

DATASET_BASE_PATH_PLANTS="datasets/train/plants_3k"
DATASET_METADATA_PATH_PLANTS="datasets/train/plants_3k.csv"

P_MASK_OUT_MASSES=0.5
P_MASK_OUT_DIRECT_FORCE=0.5
P_MASK_OUT_INDIRECT_FORCE=0.5

accelerate launch \
  --config_file scripts/accelerate/accelerate_config_4_gpu_zero_stage_2_offload_optimizer.yaml \
  scripts/train/train.py \
  --dataset_base_path ${DATASET_BASE_PATH_BALLS} ${DATASET_BASE_PATH_DOMINOS} ${DATASET_BASE_PATH_PLANTS} \
  --dataset_metadata_path ${DATASET_METADATA_PATH_BALLS} ${DATASET_METADATA_PATH_DOMINOS} ${DATASET_METADATA_PATH_PLANTS} \
  --control_signal_type ${CONTROL_SIGNAL_TYPE} \
  --controlnet_num_layers ${CONTROLNET_NUM_LAYERS} \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --save_steps 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models "controlnet" \
  --output_path "outputs/${CONTROL_SIGNAL_TYPE}" \
  --extra_inputs "input_image" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0 \
  --max_grad_norm 1 \
  --p_mask_out_masses ${P_MASK_OUT_MASSES} \
  --p_mask_out_direct_force ${P_MASK_OUT_DIRECT_FORCE} \
  --p_mask_out_indirect_force ${P_MASK_OUT_INDIRECT_FORCE} \
  --wandb_logging \
  --controlnet_checkpoint ${CKPT_PATH}