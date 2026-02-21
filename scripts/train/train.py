import torch, os
from src.goal_force.wan_video_new import WanVideoPipeline, ModelConfig
from src.goal_force.utils import DiffusionTrainingModule, launch_training_task, wan_parser
from src.goal_force.unified_dataset import (
    ControlSignalDataset_CannyEdge, 
    ControlSignalDataset_Balls,
    ControlSignalDataset_Dominos,
    ControlSignalDataset_Plants,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        control_signal_type=None,
        controlnet_num_layers=0,
        controlnet_checkpoint=None,
        controlnet_stride=None,
        apply_strided_controlnet=False,
        offline_load=False,
    ):
        super().__init__()
        # Load models
        implemented_controlnet_types = [
            "canny_edge", 
            "direct_force_and_goal_force_and_mass"
        ]
        if control_signal_type not in implemented_controlnet_types:
            raise NotImplementedError
        controlnet = control_signal_type in implemented_controlnet_types

        if offline_load:
            print(f"Training using offline model paths...")
            model_configs = self.parse_model_configs_offline(model_paths, model_id_with_origin_paths, enable_fp8_training=False)

            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, 
                controlnet=controlnet, controlnet_num_layers=controlnet_num_layers, 
                controlnet_stride=controlnet_stride, apply_strided_controlnet=apply_strided_controlnet,
                tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*", path="./models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl") # override tokenizer_config with path for offline training
                )
        else:
            model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)

            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, 
                controlnet=controlnet, controlnet_num_layers=controlnet_num_layers, 
                controlnet_stride=controlnet_stride, apply_strided_controlnet=apply_strided_controlnet)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
            control_signal_type=control_signal_type,
            controlnet_checkpoint=controlnet_checkpoint
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            # control video, for controlnet module
            "control_signal_video": data["control_video"],
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

def get_dataset(args):

    if args.control_signal_type == "canny_edge":

        # for canny edge
        dataset = ControlSignalDataset_CannyEdge(
            base_path=args.dataset_base_path[0],
            metadata_path=args.dataset_metadata_path[0],
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=ControlSignalDataset_CannyEdge.default_video_operator(
                base_path=args.dataset_base_path[0],
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
        )

    elif args.control_signal_type == "direct_force_and_goal_force_and_mass":

        # balls
        dataset_balls = ControlSignalDataset_Balls(
            base_path=args.dataset_base_path[0],
            metadata_path=args.dataset_metadata_path[0],
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            is_validation_dataset=False,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            p_mask_out_masses=args.p_mask_out_masses,
            p_mask_out_direct_force=args.p_mask_out_direct_force,
            p_mask_out_indirect_force=args.p_mask_out_indirect_force,
        )

        # dominos
        dataset_dominos = ControlSignalDataset_Dominos(
            base_path=args.dataset_base_path[1],
            metadata_path=args.dataset_metadata_path[1],
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            is_validation_dataset=False,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            p_mask_out_masses=args.p_mask_out_masses,
            p_mask_out_direct_force=args.p_mask_out_direct_force,
            p_mask_out_indirect_force=args.p_mask_out_indirect_force,
        )

        # plants
        dataset_plants = ControlSignalDataset_Plants(
            base_path=args.dataset_base_path[2],
            metadata_path=args.dataset_metadata_path[2],
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            is_validation_dataset=False,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
        )

        dataset = torch.utils.data.ConcatDataset([dataset_balls, dataset_dominos, dataset_plants])

    else:
        raise NotImplementedError

    return dataset

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = get_dataset(args)

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        control_signal_type=args.control_signal_type,
        controlnet_checkpoint=args.controlnet_checkpoint,
        controlnet_num_layers=args.controlnet_num_layers,
        controlnet_stride=args.controlnet_stride,
        apply_strided_controlnet=args.apply_strided_controlnet,
        offline_load=args.offline_load
    )

    launch_training_task(dataset, model, args=args)
