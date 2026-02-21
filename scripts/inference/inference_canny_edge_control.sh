DEVICE_ID=0
WORLD_SIZE=1
SEED=5
CONTROL_SIGNAL_TYPE="canny_edge"
MODEL_CKPT_PATH="checkpoints/wan2.2_controlnet_canny_edge/step-500.safetensors"

# CSV file paths for canny edge inference examples
EXAMPLE_PATHS=(
  "datasets/examples/canny-edge-examples/pixabay_World_Heritage_Sites_41371_007.csv"
  "datasets/examples/canny-edge-examples/pexels_landscape_landscape_5259731_005.csv"
)

python scripts/inference/inference_canny_edge_control.py \
  --device_id ${DEVICE_ID} \
  --world_size ${WORLD_SIZE} \
  --seed ${SEED} \
  --control_signal_type ${CONTROL_SIGNAL_TYPE} \
  --model_ckpt_path ${MODEL_CKPT_PATH} \
  --example_paths "${EXAMPLE_PATHS[@]}"
