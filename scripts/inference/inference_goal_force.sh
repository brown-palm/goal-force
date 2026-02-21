DEVICE_ID=0
WORLD_SIZE=1
SEED=5
CONTROL_SIGNAL_TYPE="goal_force"
MODEL_CKPT_PATH="checkpoints/goal_force/step-3000.safetensors"

# CSV file paths for inference examples
EXAMPLE_PATHS=(
  "datasets/examples/human-object-interaction/_bulb_tool_obj1_prompt1.csv"
  "datasets/examples/human-object-interaction/_toycar_obj1_prompt1.csv"
  "datasets/examples/tool-object-interaction/_pool_tool_obj1_prompt1.csv"
  "datasets/examples/animal-object-interaction/_paw_tool2_obj1_prompt1.csv"
)

python scripts/inference/inference_goal_force.py \
  --device_id ${DEVICE_ID} \
  --world_size ${WORLD_SIZE} \
  --seed ${SEED} \
  --control_signal_type ${CONTROL_SIGNAL_TYPE} \
  --model_ckpt_path ${MODEL_CKPT_PATH} \
  --example_paths "${EXAMPLE_PATHS[@]}"
