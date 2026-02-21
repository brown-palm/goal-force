import torch, torchvision, imageio, os, json, pandas
import numpy as np
import imageio.v3 as iio
from PIL import Image
from tqdm import tqdm
from controlnet_aux import CannyDetector
import pickle
import cv2
import glob
import math
from torchvision import transforms
from einops import rearrange
from typing import List


def load_video_to_pil(video_path: str) -> List[Image.Image]:
    """
    Loads a video file from the given path and returns a list of its frames
    as PIL Image objects.

    Args:
        video_path (str): The file path to the video.

    Returns:
        List[Image.Image]: A list of frames as PIL Images.

    Raises:
        IOError: If the video file cannot be opened or found.
    """
    
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video file at {video_path}")

    pil_frames: List[Image.Image] = []
    
    try:
        while True:
            # 2. Read one frame at a time
            # ret is a boolean: True if a frame was read, False otherwise
            # frame is the NumPy array (in BGR format)
            ret, frame = cap.read()

            if not ret:
                # If ret is False, we've reached the end of the video
                break

            # 3. Convert frame from BGR (OpenCV) to RGB (PIL)
            # This is a crucial step!
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 4. Convert the NumPy array to a PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # 5. Add the PIL Image to the list
            pil_frames.append(pil_image)
    
    finally:
        # 6. Release the video capture object
        # This is done in a 'finally' block to ensure it happens
        # even if an error occurs during frame reading.
        cap.release()

    return pil_frames


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        try:
            # This is the line that can fail
            reader = imageio.get_reader(data)
            num_frames = self.get_num_frames(reader)
            frames = []
            for frame_id in range(num_frames):
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)
            reader.close()
            return frames
        except Exception as e:
            # If any error occurs, log it and return None instead of crashing
            print(f"WARNING: Skipping corrupted or unreadable video file: {data}. Error: {e}")
            return None



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key]
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True


class ControlSignalDataset_CannyEdge(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.data_cache_location = "datasets/OpenVid-1M/OpenVid-1M-cache.pkl"
        self.load_metadata(metadata_path)
        self.canny_detector = CannyDetector()

    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        if path is None:
            print("self.base_path is None, no search needed.")
            return None
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:

            print("The OpenVid-1M CSV lists more mp4s than you probably downloaded.")
            print("We're filtering thru them now to find the ones in the videos folder...")
            
            # Check if cache has already been computed. If so, verify that it contains valid videos.
            cache_has_been_computed = os.path.exists(self.data_cache_location)
            if cache_has_been_computed:

                print(f"... using previously found list, which was saved to {self.data_cache_location}.")
                print("If you want to re-compute, delete that cache and run again.")

                # load data from cache
                with open(self.data_cache_location, 'rb') as f:
                    uncached_data = pickle.load(f)
                self.data = uncached_data
                print("Data has been loaded from cache!")
                
            elif not cache_has_been_computed:
                # If cache hasn't already been computed, compute and save it
                metadata = pandas.read_csv(metadata_path)
                # filter for video files that are in the directory
                self.data = []
                for i in tqdm(range(len(metadata))):
                    row_dict = metadata.iloc[i].to_dict()
                    video_path = os.path.join(self.base_path, row_dict["video"])
                    this_video_file_actually_exists = os.path.exists(video_path)
                    if this_video_file_actually_exists:
                        self.data.append(row_dict)
                # Save the data to a file
                with open(self.data_cache_location, 'wb') as f:
                    pickle.dump(self.data, f)
                print("Data has been cached to data_cache.pkl !")



    # ========================================================================
    # 2. NEW VALIDATION METHOD: It takes a video path and prompt directly.
    # ========================================================================
    def process_for_validation(self, video_path: str, prompt: str):
        """
        Processes a single video path and text prompt for validation/inference.
        
        Args:
            video_path (str): The file path to the input video.
            prompt (str): The text prompt.
            
        Returns:
            dict: A dictionary containing the 'prompt' and the processed 'control_video' tensor.
        """
        # Step 1: Use the existing video operator to load and process the video
        # This ensures validation data is processed identically to training data.
        processed_video = self.main_data_operator(video_path)
        
        # Step 2: Generate the Canny edge control video using the helper method
        control_video = self._generate_control_video(processed_video)
        
        # Step 3: Return the data in the expected dictionary format
        return {
            "prompt": prompt,
            "control_signal_video": control_video
        }

    # ========================================================================
    # 1. NEW HELPER METHOD: Extracts the Canny edge generation logic.
    #    This can now be used by both training (`__getitem__`) and validation.
    # ========================================================================
    def _generate_control_video(self, processed_video):
        """
        Generates a Canny edge control video from a pre-processed video tensor/list.
        """
        video_as_np = np.array(processed_video) # (49, 480, 832, 3), uint8's in [0, 255]

        canny_frames = []
        for frame in video_as_np:
            canny = self.canny_detector(frame) # (512, 896, 3)
            if canny.shape[:2] != frame.shape[:2]:
                canny = cv2.resize(canny, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

            canny_frames.append(canny)

        canny_frames_np = np.stack(canny_frames, axis=0)  # (49, 480, 832, 3)
        canny_frames_torch = torch.from_numpy(canny_frames_np).to(torch.float32) / 127.5 - 1.0
        return canny_frames_torch.to(torch.bfloat16)



    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            # print("data_id:", data_id)
            data = self.data[data_id % len(self.data)].copy()

            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key]
                    elif key in self.data_file_keys:
                        processed_data = self.main_data_operator(data[key])
                        # Check if the data operator returned None (i.e., failed to load)
                        if processed_data is None:
                            return None # Mark this entire sample as invalid
                        data[key] = processed_data
            # This code only runs if the video was loaded successfully
            data["control_video"] = self._generate_control_video(data["video"])

            # replace "caption" with "prompt"
            data["prompt"] = data.pop("caption")

        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True








# keep
class ControlSignalDataset_Balls(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        is_validation_dataset=False,
        num_frames=None,
        height=None,
        width=None,
        p_mask_out_direct_force=0.0,
        p_mask_out_indirect_force=0.0,
        p_mask_out_masses=0.0,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None

        self.is_validation_dataset = is_validation_dataset
        self.num_frames=num_frames
        self.height = height
        self.width = width

        self.p_mask_out_direct_force = p_mask_out_direct_force
        self.p_mask_out_indirect_force = p_mask_out_indirect_force
        # these are not independent events!
        assert self.p_mask_out_direct_force + self.p_mask_out_indirect_force <= 1 
        self.p_mask_out_masses = p_mask_out_masses
        assert 0.0 <= self.p_mask_out_masses <= 1.0

        self.to_tensor_transform = transforms.ToTensor()
        self.to_pil_transform = transforms.ToPILImage()
        if self.is_validation_dataset:
            self.media_type = "image"
            self.blob_ext =  "*.png"
        else:
            self.media_type = "video"
            self.blob_ext = "*.mp4"

        self.load_metadata()
        # self.debug_dataloader() # uncomment this out when debugging dataloader!

    def debug_dataloader(self):

        data_id = 0

        (
            pixel_values,           # (81, 3, 480, 832)
            caption,                # str
            projectile_force,       # np.float64
            projectile_angle,       # np.float64
            projectile_x_pos,       # np.float64
            projectile_y_pos,       # np.float64
            target_indirect_force,  # np.float64
            target_indirect_angle,  # np.float64
            target_x_pos,           # np.float64
            target_y_pos,           # np.float64
            file_id,                # str
            masses,                 # dict
            coords,                 # dict
        ) = self.get_batch(data_id % len(self.df))

        control_video = self._generate_control_video(
            projectile_force, projectile_angle, projectile_x_pos, projectile_y_pos, 
            target_indirect_force, target_indirect_angle, target_x_pos, target_y_pos, 
            num_frames=self.num_frames, 
            num_channels=3, height=self.height, width=self.width,
            masses=masses, coords=coords) 

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        if path is None:
            print("self.base_path is None, no search needed.")
            return None
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self):

        if not self.is_validation_dataset:
            file_paths = glob.glob(os.path.join(self.base_path, self.blob_ext))
            file_names = set([os.path.basename(x) for x in file_paths]) # list of videos or images...
            self.df = pandas.read_csv(self.metadata_path)

            # only keep the rows in the csv whose videos we can find
            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == True]

            self.min_force = float(self.df["projectile_force_magnitude"].min())
            self.max_force = float(self.df["projectile_force_magnitude"].max())
            print(f"min_force = {self.min_force}, max_force = {self.max_force}")

            indirect_forces = self.df[self.df["target_indirect_force_magnitude"] > -1]
            self.min_indirect_force = float(indirect_forces["target_indirect_force_magnitude"].min())
            self.max_indirect_force = float(indirect_forces["target_indirect_force_magnitude"].max())

            self.min_mass = float(self.df["projectile_mass"].min())
            self.max_mass = float(self.df["projectile_mass"].max())
            print(f"min_mass = {self.min_mass}, max_mass = {self.max_mass}")

            self.length = self.df.shape[0]
        elif self.is_validation_dataset:
            file_paths = glob.glob(os.path.join(self.base_path, "images", self.blob_ext))
            file_names = set([os.path.basename(x) for x in file_paths])
            self.df = pandas.read_csv(self.metadata_path)

            # only keep the rows in the csv whose videos we can find
            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == True]

            self.min_force = 0.0
            self.max_force = 1.0

    def _generate_control_video(self,
        force, angle, x_pos, y_pos, 
        target_indirect_force, target_indirect_angle, target_x_pos, target_y_pos, 
        num_frames=49, num_channels=3, height=480, width=720, masses={}, coords={}
    ):
        # STEP 0: initialize the tensor of all zeros...
        controlnet_signal = torch.zeros((num_frames, num_channels, height, width)) # (49, 3, 480, 720)

        # STEP 1: mask out none or one of the direct force and the indirect force, but not both!
        if force == -1:
            # If the original force is not specified, then we mask out this channel.
            # This happens in our inference data, for example.
            mask_out_direct_force = True
            mask_out_indirect_force = False
        elif target_indirect_force == -1:
            # if there are no collisions, we mask out the (non-existent) indirect force,
            # and we never mask out the direct force. there's no target indirect force if it's NO collision!!
            mask_out_direct_force = False
            mask_out_indirect_force = True
        else:
            mask_out_direct_force = False
            mask_out_indirect_force = False
            unif_random_sample = np.random.uniform(low=0.0, high=1.0)
            if unif_random_sample < self.p_mask_out_direct_force:
                mask_out_direct_force = True
            elif self.p_mask_out_direct_force <= unif_random_sample <= self.p_mask_out_direct_force + self.p_mask_out_indirect_force:
                mask_out_indirect_force = True

        DISPLACEMENT_FOR_MAX_FORCE = width / 2
        DISPLACEMENT_FOR_MIN_FORCE = width / 8

        # STEP 2: gaussian blob for direct force in first channel...
        if not mask_out_direct_force:
            x_pos_start = x_pos*width
            y_pos_start = (1-y_pos)*height

            force_percent = (force - self.min_force) / (self.max_force - self.min_force)
            total_displacement = DISPLACEMENT_FOR_MIN_FORCE + (DISPLACEMENT_FOR_MAX_FORCE - DISPLACEMENT_FOR_MIN_FORCE) * force_percent

            x_pos_end = x_pos_start + total_displacement * math.cos(angle * torch.pi / 180.0)
            y_pos_end = y_pos_start - total_displacement * math.sin(angle * torch.pi / 180.0)

            for frame in range(num_frames):
                t = frame / (num_frames-1)
                x_pos_ = x_pos_start * (1-t) + x_pos_end * t # t = 0 --> start; t = 0 --> end
                y_pos_ = y_pos_start * (1-t) + y_pos_end * t # t = 0 --> start; t = 0 --> end
                blob_tensor = self.get_gaussian_blob(x=x_pos_, y=y_pos_, radius=20, amplitude=1.0, shape=(1, height, width))[0]
                controlnet_signal[:, 0][frame] += blob_tensor

        # STEP 3: gaussian blob for indirect force in second channel
        if not mask_out_indirect_force:
            x_pos_start = target_x_pos*width
            y_pos_start = (1-target_y_pos)*height
            force_percent = (target_indirect_force - self.min_indirect_force) / (self.max_indirect_force - self.min_indirect_force)
            total_displacement = DISPLACEMENT_FOR_MIN_FORCE + (DISPLACEMENT_FOR_MAX_FORCE - DISPLACEMENT_FOR_MIN_FORCE) * force_percent

            x_pos_end = x_pos_start + total_displacement * math.cos(target_indirect_angle * torch.pi / 180.0)
            y_pos_end = y_pos_start - total_displacement * math.sin(target_indirect_angle * torch.pi / 180.0)

            for frame in range(num_frames):
                t = frame / (num_frames-1)
                x_pos_ = x_pos_start * (1-t) + x_pos_end * t # t = 0 --> start; t = 0 --> end
                y_pos_ = y_pos_start * (1-t) + y_pos_end * t # t = 0 --> start; t = 0 --> end
                blob_tensor = self.get_gaussian_blob(x=x_pos_, y=y_pos_, radius=20, amplitude=1.0, shape=(1, height, width))[0]
                controlnet_signal[:, 1][frame] += blob_tensor
        
        # STEP 4: rearrange...
        controlnet_signal = rearrange(controlnet_signal, 'f c h w -> f h w c') # (49, 3, 480, 832) --> (49, 480, 832, 3)

        # STEP 5: blobs with radius proportional to masses in third channel

        # we overwrite the third frame to be all zeros, regardless of whether we end up
        # encoding the mass in the third channel...
        controlnet_signal[:, :, :, 2] = 0

        # with likelihood self.p_mask_out_masses, we set the third channel (the mass channel) to be all zeros
        mask_out_masses = np.random.uniform(low=0.0, high=1.0) < self.p_mask_out_masses
        if not mask_out_masses:
        
            # STEP 2A: make blob at projectile center
            xpos_projectile = coords["projectile"][0]
            ypos_projectile = height - coords["projectile"][1]
            mass_projectile = masses["projectile"]
            if mass_projectile > -1:
                blob_mass_projectile = self.get_blob_for_mass(
                    xpos_projectile, ypos_projectile, mass_projectile, shape=(num_frames, height, width)
                )
                controlnet_signal[:, :, :, 2] += blob_mass_projectile

            # STEP 2B: make blob at target center
            xpos_target = coords["target"][0]
            ypos_target = height - coords["target"][1]
            mass_target = masses["target"]
            if mass_target > -1:
                # if mass_target is -1, then this is a "no collision" sample, so we shouldn't plot it...
                blob_mass_target = self.get_blob_for_mass(
                    xpos_target, ypos_target, mass_target, shape=(num_frames, height, width)
                )
                controlnet_signal[:, :, :, 2] += blob_mass_target

            # STEP 2C: make blobs at distractors' centers
            for mass_distractor, (xpos_distractor, ypos_distractor) in zip(masses["distractors"], coords["distractors"]):

                if mass_distractor == -1: # this filter might not be necessary, but we include here anyway
                    continue
                
                blob_mass_distractor = self.get_blob_for_mass(
                    xpos_distractor, height - ypos_distractor, mass_distractor, shape=(num_frames, height, width)
                )
                controlnet_signal[:, :, :, 2] += blob_mass_distractor

            # STEP 2D: clip it to make sure its max is 1...
            controlnet_signal = torch.clamp(controlnet_signal, min=0.0, max=1.0)
        
        return controlnet_signal.to(torch.bfloat16)

    def get_blob_for_mass(self, xpos, ypos, mass, shape=None):

        # hparams to set! i set them to this cuz the moving blob has radius 20
        min_blob_mass_radius = 5
        max_blob_mass_radius = 40

        t = (mass - self.min_mass) / (self.max_mass - self.min_mass)
        blob_radius = (1-t)*min_blob_mass_radius + t * max_blob_mass_radius
        blob_mass = self.get_gaussian_blob(x=xpos, y=ypos, radius=blob_radius, amplitude=1.0, shape=shape)

        return blob_mass

    def get_gaussian_blob(self, x, y, radius=10, amplitude=1.0, shape=(3, 480, 720), device=None):
        """
        Create a tensor containing a Gaussian blob at the specified location.
        
        Args:
            x (int): x-coordinate of the blob center
            y (int): y-coordinate of the blob center
            radius (int, optional): Radius of the Gaussian blob. Defaults to 10.
            amplitude (float, optional): Maximum intensity of the blob. Defaults to 1.0.
            shape (tuple, optional): Shape of the output tensor (channels, height, width). Defaults to (3, 480, 720).
            device (torch.device, optional): Device to create the tensor on. Defaults to None.
        
        Returns:
            torch.Tensor: Tensor of shape (channels, height, width) containing the Gaussian blob
        """
        num_channels, height, width = shape
        
        # Create a new tensor filled with zeros
        blob_tensor = torch.zeros(shape, device=device)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Calculate squared distance from (x, y)
        squared_dist = (x_grid - x) ** 2 + (y_grid - y) ** 2
        
        # Create Gaussian blob using the squared distance
        gaussian = amplitude * torch.exp(-squared_dist / (2.0 * radius ** 2))
        
        # Add the Gaussian blob to all channels
        for c in range(num_channels):
            blob_tensor[c] = gaussian
        
        return blob_tensor

    def get_batch(self, idx):

        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force = item['projectile_force_magnitude']
        angle = item['projectile_force_angle']

        target_indirect_force_angle     = item["target_indirect_force_angle"]
        target_indirect_force_magnitude = item["target_indirect_force_magnitude"]
        target_x_pos                    = item["target_coordx"] / item["width"]
        target_y_pos                    = item["target_coordy"] / item["height"]
        
        if self.media_type == "image":
            file_path = os.path.join(self.base_path, "images", file_name)

            image = Image.open(file_path)
            desired_size = (self.width, self.height)
            if image.size != desired_size:
                print("Resizing image.")
                image = image.resize(desired_size, resample=Image.Resampling.LANCZOS)
            pixel_values = self.to_tensor_transform(image)
            pixel_values = 2*pixel_values - 1

            file_id = file_name.split(".png")[0]
            x_pos = item["projectile_coordx"] / item["width"]
            y_pos = item["projectile_coordy"] / item["height"]

            masses = {
                "projectile"    : item["projectile_mass"],
                "target"        : item["target_mass"],
                "distractors"   : []
            }
            coords = {
                "projectile"    : [int(item["projectile_coordx"]), int(item["projectile_coordy"])],
                "target"        : [int(item["target_coordx"]), int(item["target_coordy"])],
                "distractors"   : []
            }

        elif self.media_type == "video":

            file_path = os.path.join(self.base_path, file_name)

            # this manually loads the frames I want...
            pixel_values = load_video_to_pil(file_path) # [<PIL.Image.Image image mode=RGB size=832x480 at 0x7F8E842DC050>, ...], len=182 (num frames)
            pixel_values = pixel_values[::2][-self.num_frames:] # len = 81
            pixel_values = torch.stack([self.to_tensor_transform(image) for image in pixel_values])  # (81, 3, 480, 832) of torch.float32 in [0, 1]
            pixel_values = 2*pixel_values - 1

            file_id = file_name.split(".mp4")[0]

            x_pos = item["projectile_coordx"] / item["width"]
            y_pos = item["projectile_coordy"] / item["height"]

            masses = {
                "projectile"    : item["projectile_mass"],
                "target"        : item["target_mass"],
                "distractors"   : []
            }
            coords = {
                "projectile"    : [int(item["projectile_coordx"]), int(item["projectile_coordy"])],
                "target"        : [int(item["target_coordx"]), int(item["target_coordy"])],
                "distractors"   : []
            }

            # confirm that the hard coded max num distractors actually corresponds to
            # what's in the video metadata csv...
            MAX_NUM_DISTRACTORS = 8
            assert f"distractor_{MAX_NUM_DISTRACTORS-1}_mass" in item
            assert f"distractor_{MAX_NUM_DISTRACTORS}_mass" not in item
            for distractor_idx in range(MAX_NUM_DISTRACTORS):

                mass    = float(item[f"distractor_{distractor_idx}_mass"])
                coordx  = int(item[f"distractor_{distractor_idx}_coordx"])
                coordy  = int(item[f"distractor_{distractor_idx}_coordy"])

                # the one with a mass and coords of -1 is the target, or else doesn't
                # correspond to a distractor! we ignore that one
                if mass == -1:
                    continue
                
                masses["distractors"].append(mass)
                coords["distractors"].append((coordx, coordy))

        return pixel_values, caption, force, angle, x_pos, y_pos, target_indirect_force_magnitude, target_indirect_force_angle, target_x_pos, target_y_pos, file_id, masses, coords


    def __getitem__(self, data_id):

        (
            pixel_values,           # (81, 3, 480, 832)
            caption,                # str
            projectile_force,       # np.float64
            projectile_angle,       # np.float64
            projectile_x_pos,       # np.float64
            projectile_y_pos,       # np.float64
            target_indirect_force,  # np.float64
            target_indirect_angle,  # np.float64
            target_x_pos,           # np.float64
            target_y_pos,           # np.float64
            file_id,                # str
            masses,                 # dict
            coords,                 # dict
        ) = self.get_batch(data_id % len(self.df))

        control_video = self._generate_control_video(
            projectile_force, projectile_angle, projectile_x_pos, projectile_y_pos, 
            target_indirect_force, target_indirect_angle, target_x_pos, target_y_pos, 
            num_frames=self.num_frames, 
            num_channels=3, height=self.height, width=self.width,
            masses=masses, coords=coords) 

        # convert the pixel values back to a PIL image list, for compatibility
        pixel_values = (pixel_values + 1) / 2 # make values in [0,1]
        if not self.is_validation_dataset:
            # during training, the tensor is a list of image frames
            pil_image_list = [self.to_pil_transform(tensor) for tensor in pixel_values]
        else:
            # during training, the tensor just a single image frame
            pil_image_list = [self.to_pil_transform(pixel_values)]

        data = {
            "video": pil_image_list,
            "prompt": caption,
            "control_video": control_video,
            "force": projectile_force,
            "angle": projectile_angle,
            "x_pos": projectile_x_pos,
            "y_pos": projectile_y_pos,
            "target_indirect_force": target_indirect_force,
            "target_indirect_angle": target_indirect_angle,
            "target_x_pos": target_x_pos,
            "target_y_pos": target_y_pos,
            "file_id": file_id,
            "masses" : masses,
            "coords" : coords,
        }

        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.df) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True


# keep
class ControlSignalDataset_Dominos(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        is_validation_dataset=False,
        num_frames=None,
        height=None,
        width=None,
        p_mask_out_direct_force=0.0,
        p_mask_out_indirect_force=0.0,
        p_mask_out_masses=0.0,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None

        self.is_validation_dataset = is_validation_dataset
        self.num_frames=num_frames
        self.height = height
        self.width = width

        self.p_mask_out_direct_force = p_mask_out_direct_force
        self.p_mask_out_indirect_force = p_mask_out_indirect_force
        # these are not independent events!
        assert self.p_mask_out_direct_force + self.p_mask_out_indirect_force <= 1 
        self.p_mask_out_masses = p_mask_out_masses
        assert 0.0 <= self.p_mask_out_masses <= 1.0

        self.to_tensor_transform = transforms.ToTensor()
        self.to_pil_transform = transforms.ToPILImage()
        if self.is_validation_dataset:
            self.media_type = "image"
            self.blob_ext =  "*.png"
        else:
            self.media_type = "video"
            self.blob_ext = "*.mp4"

        self.load_metadata()
        # self.debug_dataloader() # uncomment this out when debugging dataloader!

    def debug_dataloader(self):

        data_id = 0

        (
            pixel_values,           # (81, 3, 480, 832)
            caption,                # str
            projectile_force,       # np.float64
            projectile_angle,       # np.float64
            projectile_x_pos,       # np.float64
            projectile_y_pos,       # np.float64
            target_indirect_force,  # np.float64
            target_indirect_angle,  # np.float64
            target_x_pos,           # np.float64
            target_y_pos,           # np.float64
            file_id,                # str
            masses,                 # dict
            coords,                 # dict
        ) = self.get_batch(data_id % len(self.df))

        control_video = self._generate_control_video(
            projectile_force, projectile_angle, projectile_x_pos, projectile_y_pos, 
            target_indirect_force, target_indirect_angle, target_x_pos, target_y_pos, 
            num_frames=self.num_frames, 
            num_channels=3, height=self.height, width=self.width,
            masses=masses, coords=coords) 

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        if path is None:
            print("self.base_path is None, no search needed.")
            return None
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self):

        if not self.is_validation_dataset:
            file_paths = glob.glob(os.path.join(self.base_path, self.blob_ext))
            file_names = set([os.path.basename(x) for x in file_paths]) # list of videos or images...
            self.df = pandas.read_csv(self.metadata_path)

            # only keep the rows in the csv whose videos we can find
            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == True]

            self.min_force = float(self.df["projectile_force_magnitude"].min())
            self.max_force = float(self.df["projectile_force_magnitude"].max())
            print(f"min_force = {self.min_force}, max_force = {self.max_force}")

            indirect_forces = self.df[self.df["target_indirect_force_magnitude"] > -1]
            self.min_indirect_force = float(indirect_forces["target_indirect_force_magnitude"].min())
            self.max_indirect_force = float(indirect_forces["target_indirect_force_magnitude"].max())

            self.min_mass = float(self.df["projectile_mass"].min())
            self.max_mass = float(self.df["projectile_mass"].max())
            print(f"min_mass = {self.min_mass}, max_mass = {self.max_mass}")

            self.length = self.df.shape[0]
        elif self.is_validation_dataset:
            file_paths = glob.glob(os.path.join(self.base_path, "images", self.blob_ext))
            file_names = set([os.path.basename(x) for x in file_paths])
            self.df = pandas.read_csv(self.metadata_path)

            # only keep the rows in the csv whose videos we can find
            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == True]

            self.min_force = 0.0
            self.max_force = 1.0

    def _generate_control_video(self,
        force, angle, x_pos, y_pos, 
        target_indirect_force, target_indirect_angle, target_x_pos, target_y_pos, 
        num_frames=49, num_channels=3, height=480, width=720, masses={}, coords={}
    ):
        # STEP 0: initialize the tensor of all zeros...
        controlnet_signal = torch.zeros((num_frames, num_channels, height, width)) # (49, 3, 480, 720)

        # STEP 1: mask out none or one of the direct force and the indirect force, but not both!
        if force == -1:
            # If the original force is not specified, then we mask out this channel.
            # This happens in our inference data, for example.
            mask_out_direct_force = True
            mask_out_indirect_force = False
        elif target_indirect_force == -1:
            # if there are no collisions, we mask out the (non-existent) indirect force,
            # and we never mask out the direct force. there's no target indirect force if it's NO collision!!
            mask_out_direct_force = False
            mask_out_indirect_force = True
        else:
            mask_out_direct_force = False
            mask_out_indirect_force = False
            unif_random_sample = np.random.uniform(low=0.0, high=1.0)
            if unif_random_sample < self.p_mask_out_direct_force:
                mask_out_direct_force = True
            elif self.p_mask_out_direct_force <= unif_random_sample <= self.p_mask_out_direct_force + self.p_mask_out_indirect_force:
                mask_out_indirect_force = True

        DISPLACEMENT_FOR_MAX_FORCE = width / 2
        DISPLACEMENT_FOR_MIN_FORCE = width / 8

        # STEP 2: gaussian blob for direct force in first channel...
        if not mask_out_direct_force:
            x_pos_start = x_pos*width
            y_pos_start = (1-y_pos)*height

            force_percent = (force - self.min_force) / (self.max_force - self.min_force)
            total_displacement = DISPLACEMENT_FOR_MIN_FORCE + (DISPLACEMENT_FOR_MAX_FORCE - DISPLACEMENT_FOR_MIN_FORCE) * force_percent

            x_pos_end = x_pos_start + total_displacement * math.cos(angle * torch.pi / 180.0)
            y_pos_end = y_pos_start - total_displacement * math.sin(angle * torch.pi / 180.0)

            for frame in range(num_frames):
                t = frame / (num_frames-1)
                x_pos_ = x_pos_start * (1-t) + x_pos_end * t # t = 0 --> start; t = 0 --> end
                y_pos_ = y_pos_start * (1-t) + y_pos_end * t # t = 0 --> start; t = 0 --> end
                blob_tensor = self.get_gaussian_blob(x=x_pos_, y=y_pos_, radius=20, amplitude=1.0, shape=(1, height, width))[0]
                controlnet_signal[:, 0][frame] += blob_tensor

        # STEP 3: gaussian blob for indirect force in second channel
        if not mask_out_indirect_force:
            x_pos_start = target_x_pos*width
            y_pos_start = (1-target_y_pos)*height
            force_percent = (target_indirect_force - self.min_indirect_force) / (self.max_indirect_force - self.min_indirect_force)
            total_displacement = DISPLACEMENT_FOR_MIN_FORCE + (DISPLACEMENT_FOR_MAX_FORCE - DISPLACEMENT_FOR_MIN_FORCE) * force_percent

            x_pos_end = x_pos_start + total_displacement * math.cos(target_indirect_angle * torch.pi / 180.0)
            y_pos_end = y_pos_start - total_displacement * math.sin(target_indirect_angle * torch.pi / 180.0)

            for frame in range(num_frames):
                t = frame / (num_frames-1)
                x_pos_ = x_pos_start * (1-t) + x_pos_end * t # t = 0 --> start; t = 0 --> end
                y_pos_ = y_pos_start * (1-t) + y_pos_end * t # t = 0 --> start; t = 0 --> end
                blob_tensor = self.get_gaussian_blob(x=x_pos_, y=y_pos_, radius=20, amplitude=1.0, shape=(1, height, width))[0]
                controlnet_signal[:, 1][frame] += blob_tensor
        
        # STEP 4: rearrange...
        controlnet_signal = rearrange(controlnet_signal, 'f c h w -> f h w c') # (49, 3, 480, 832) --> (49, 480, 832, 3)

        # STEP 5: blobs with radius proportional to masses in third channel

        # we overwrite the third frame to be all zeros, regardless of whether we end up
        # encoding the mass in the third channel...
        controlnet_signal[:, :, :, 2] = 0

        # with likelihood self.p_mask_out_masses, we set the third channel (the mass channel) to be all zeros
        mask_out_masses = np.random.uniform(low=0.0, high=1.0) < self.p_mask_out_masses
        if not mask_out_masses:
        
            # STEP 2A: make blob at projectile center
            xpos_projectile = coords["projectile"][0]
            ypos_projectile = height - coords["projectile"][1]
            mass_projectile = masses["projectile"]
            if mass_projectile > -1:
                blob_mass_projectile = self.get_blob_for_mass(
                    xpos_projectile, ypos_projectile, mass_projectile, shape=(num_frames, height, width)
                )
                controlnet_signal[:, :, :, 2] += blob_mass_projectile

            # STEP 2B: make blob at target center
            xpos_target = coords["target"][0]
            ypos_target = height - coords["target"][1]
            mass_target = masses["target"]
            if mass_target > -1:
                # if mass_target is -1, then this is a "no collision" sample, so we shouldn't plot it...
                blob_mass_target = self.get_blob_for_mass(
                    xpos_target, ypos_target, mass_target, shape=(num_frames, height, width)
                )
                controlnet_signal[:, :, :, 2] += blob_mass_target

            # STEP 2C: make blobs at distractors' centers
            for mass_distractor, (xpos_distractor, ypos_distractor) in zip(masses["distractors"], coords["distractors"]):

                if mass_distractor == -1: # this filter might not be necessary, but we include here anyway
                    continue
                
                blob_mass_distractor = self.get_blob_for_mass(
                    xpos_distractor, height - ypos_distractor, mass_distractor, shape=(num_frames, height, width)
                )
                controlnet_signal[:, :, :, 2] += blob_mass_distractor

            # STEP 2D: clip it to make sure its max is 1...
            controlnet_signal = torch.clamp(controlnet_signal, min=0.0, max=1.0)
        
        return controlnet_signal.to(torch.bfloat16)

    def get_blob_for_mass(self, xpos, ypos, mass, shape=None):

        # hparams to set! i set them to this cuz the moving blob has radius 20
        min_blob_mass_radius = 5
        max_blob_mass_radius = 40

        t = (mass - self.min_mass) / (self.max_mass - self.min_mass)
        blob_radius = (1-t)*min_blob_mass_radius + t * max_blob_mass_radius
        blob_mass = self.get_gaussian_blob(x=xpos, y=ypos, radius=blob_radius, amplitude=1.0, shape=shape)

        return blob_mass

    def get_gaussian_blob(self, x, y, radius=10, amplitude=1.0, shape=(3, 480, 720), device=None):
        """
        Create a tensor containing a Gaussian blob at the specified location.
        
        Args:
            x (int): x-coordinate of the blob center
            y (int): y-coordinate of the blob center
            radius (int, optional): Radius of the Gaussian blob. Defaults to 10.
            amplitude (float, optional): Maximum intensity of the blob. Defaults to 1.0.
            shape (tuple, optional): Shape of the output tensor (channels, height, width). Defaults to (3, 480, 720).
            device (torch.device, optional): Device to create the tensor on. Defaults to None.
        
        Returns:
            torch.Tensor: Tensor of shape (channels, height, width) containing the Gaussian blob
        """
        num_channels, height, width = shape
        
        # Create a new tensor filled with zeros
        blob_tensor = torch.zeros(shape, device=device)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Calculate squared distance from (x, y)
        squared_dist = (x_grid - x) ** 2 + (y_grid - y) ** 2
        
        # Create Gaussian blob using the squared distance
        gaussian = amplitude * torch.exp(-squared_dist / (2.0 * radius ** 2))
        
        # Add the Gaussian blob to all channels
        for c in range(num_channels):
            blob_tensor[c] = gaussian
        
        return blob_tensor

    def get_batch(self, idx):

        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force = item['projectile_force_magnitude']
        angle = item['projectile_force_angle']

        target_indirect_force_angle     = item["target_indirect_force_angle"]
        target_indirect_force_magnitude = item["target_indirect_force_magnitude"]
        target_x_pos                    = item["target_coordx"] / item["width"]
        target_y_pos                    = item["target_coordy"] / item["height"]
        
        if self.media_type == "image":
            file_path = os.path.join(self.base_path, "images", file_name)

            image = Image.open(file_path)
            desired_size = (self.width, self.height)
            if image.size != desired_size:
                print("Resizing image.")
                image = image.resize(desired_size, resample=Image.Resampling.LANCZOS)
            pixel_values = self.to_tensor_transform(image)
            pixel_values = 2*pixel_values - 1

            file_id = file_name.split(".png")[0]
            x_pos = item["projectile_coordx"] / item["width"]
            y_pos = item["projectile_coordy"] / item["height"]

            masses = {
                "projectile"    : item["projectile_mass"],
                "target"        : item["target_mass"],
                "distractors"   : []
            }
            coords = {
                "projectile"    : [int(item["projectile_coordx"]), int(item["projectile_coordy"])],
                "target"        : [int(item["target_coordx"]), int(item["target_coordy"])],
                "distractors"   : []
            }

        elif self.media_type == "video":

            file_path = os.path.join(self.base_path, file_name)

            # this manually loads the frames I want...
            pixel_values = load_video_to_pil(file_path) # [<PIL.Image.Image image mode=RGB size=832x480 at 0x7F8E842DC050>, ...], len=182 (num frames)
            pixel_values = pixel_values[14:][0:self.num_frames] # len = 81
            pixel_values = torch.stack([self.to_tensor_transform(image) for image in pixel_values])  # (81, 3, 480, 832) of torch.float32 in [0, 1]
            pixel_values = 2*pixel_values - 1

            file_id = file_name.split(".mp4")[0]

            x_pos = item["projectile_coordx"] / item["width"]
            y_pos = item["projectile_coordy"] / item["height"]

            masses = {
                "projectile"    : item["projectile_mass"],
                "target"        : item["target_mass"],
                "distractors"   : []
            }
            coords = {
                "projectile"    : [int(item["projectile_coordx"]), int(item["projectile_coordy"])],
                "target"        : [int(item["target_coordx"]), int(item["target_coordy"])],
                "distractors"   : []
            }

        return pixel_values, caption, force, angle, x_pos, y_pos, target_indirect_force_magnitude, target_indirect_force_angle, target_x_pos, target_y_pos, file_id, masses, coords


    def __getitem__(self, data_id):

        (
            pixel_values,           # (81, 3, 480, 832)
            caption,                # str
            projectile_force,       # np.float64
            projectile_angle,       # np.float64
            projectile_x_pos,       # np.float64
            projectile_y_pos,       # np.float64
            target_indirect_force,  # np.float64
            target_indirect_angle,  # np.float64
            target_x_pos,           # np.float64
            target_y_pos,           # np.float64
            file_id,                # str
            masses,                 # dict
            coords,                 # dict
        ) = self.get_batch(data_id % len(self.df))

        control_video = self._generate_control_video(
            projectile_force, projectile_angle, projectile_x_pos, projectile_y_pos, 
            target_indirect_force, target_indirect_angle, target_x_pos, target_y_pos, 
            num_frames=self.num_frames, 
            num_channels=3, height=self.height, width=self.width,
            masses=masses, coords=coords) 

        # convert the pixel values back to a PIL image list, for compatibility
        pixel_values = (pixel_values + 1) / 2 # make values in [0,1]
        if not self.is_validation_dataset:
            # during training, the tensor is a list of image frames
            pil_image_list = [self.to_pil_transform(tensor) for tensor in pixel_values]
        else:
            # during training, the tensor just a single image frame
            pil_image_list = [self.to_pil_transform(pixel_values)]

        data = {
            "video": pil_image_list,
            "prompt": caption,
            "control_video": control_video,
            "force": projectile_force,
            "angle": projectile_angle,
            "x_pos": projectile_x_pos,
            "y_pos": projectile_y_pos,
            "target_indirect_force": target_indirect_force,
            "target_indirect_angle": target_indirect_angle,
            "target_x_pos": target_x_pos,
            "target_y_pos": target_y_pos,
            "file_id": file_id,
            "masses" : masses,
            "coords" : coords,
        }

        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.df) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True


class ControlSignalDataset_Plants(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        is_validation_dataset=False,
        num_frames=None,
        height=None,
        width=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None

        self.is_validation_dataset = is_validation_dataset
        self.num_frames=num_frames
        self.height = height
        self.width = width

        self.to_tensor_transform = transforms.ToTensor()
        self.to_pil_transform = transforms.ToPILImage()
        if self.is_validation_dataset:
            self.media_type = "image"
            self.blob_ext =  "*.png"
        else:
            self.media_type = "video"
            self.blob_ext = "*.mp4"

        self.load_metadata()
        # batch = self.get_batch(0) # for debugging

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        if path is None:
            print("self.base_path is None, no search needed.")
            return None
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self):

        if not self.is_validation_dataset:
            file_paths = glob.glob(os.path.join(self.base_path, self.blob_ext))
            file_names = set([os.path.basename(x) for x in file_paths]) # list of videos or images...
            self.df = pandas.read_csv(self.metadata_path)

            # only keep the rows in the csv whose videos we can find
            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == True]

            self.min_force = float(self.df["force"].min())
            self.max_force = float(self.df["force"].max())
            print(f"min_force = {self.min_force}, max_force = {self.max_force}")

            self.length = self.df.shape[0]
        elif self.is_validation_dataset:
            file_paths = glob.glob(os.path.join(self.base_path, "images", self.blob_ext))
            file_names = set([os.path.basename(x) for x in file_paths])
            self.df = pandas.read_csv(self.metadata_path)

            # only keep the rows in the csv whose videos we can find
            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == True]

            self.min_force = 0.0
            self.max_force = 1.0

    
    def _generate_control_video(self, force, angle, x_pos, y_pos, num_frames=49, num_channels=3, height=480, width=720):

        controlnet_signal = torch.zeros((num_frames, num_channels, height, width)) # (49, 3, 480, 720)

        x_pos_start = x_pos*width
        y_pos_start = (1-y_pos)*height

        DISPLACEMENT_FOR_MAX_FORCE = width / 2
        DISPLACEMENT_FOR_MIN_FORCE = width / 8

        force_percent = (force - self.min_force) / (self.max_force - self.min_force)
        total_displacement = DISPLACEMENT_FOR_MIN_FORCE + (DISPLACEMENT_FOR_MAX_FORCE - DISPLACEMENT_FOR_MIN_FORCE) * force_percent

        x_pos_end = x_pos_start + total_displacement * math.cos(angle * torch.pi / 180.0)
        y_pos_end = y_pos_start - total_displacement * math.sin(angle * torch.pi / 180.0)

        for frame in range(num_frames):

            t = frame / (num_frames-1)
            x_pos_ = x_pos_start * (1-t) + x_pos_end * t # t = 0 --> start; t = 0 --> end
            y_pos_ = y_pos_start * (1-t) + y_pos_end * t # t = 0 --> start; t = 0 --> end

            blob_tensor = self.get_gaussian_blob(x=x_pos_, y=y_pos_, radius=20, amplitude=1.0, shape=(num_channels, height, width))

            controlnet_signal[frame] += blob_tensor

        controlnet_signal = rearrange(controlnet_signal, 'f c h w -> f h w c') # (49, 3, 480, 832) --> (49, 480, 832, 3)

        # Zero out the last two frames, since those are goal force and mass
        controlnet_signal[:, :, :, 1:3] = 0
        
        return controlnet_signal.to(torch.bfloat16)

    def get_gaussian_blob(self, x, y, radius=10, amplitude=1.0, shape=(3, 480, 720), device=None):
        """
        Create a tensor containing a Gaussian blob at the specified location.
        
        Args:
            x (int): x-coordinate of the blob center
            y (int): y-coordinate of the blob center
            radius (int, optional): Radius of the Gaussian blob. Defaults to 10.
            amplitude (float, optional): Maximum intensity of the blob. Defaults to 1.0.
            shape (tuple, optional): Shape of the output tensor (channels, height, width). Defaults to (3, 480, 720).
            device (torch.device, optional): Device to create the tensor on. Defaults to None.
        
        Returns:
            torch.Tensor: Tensor of shape (channels, height, width) containing the Gaussian blob
        """
        num_channels, height, width = shape
        
        # Create a new tensor filled with zeros
        blob_tensor = torch.zeros(shape, device=device)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Calculate squared distance from (x, y)
        squared_dist = (x_grid - x) ** 2 + (y_grid - y) ** 2
        
        # Create Gaussian blob using the squared distance
        gaussian = amplitude * torch.exp(-squared_dist / (2.0 * radius ** 2))
        
        # Add the Gaussian blob to all channels
        for c in range(num_channels):
            blob_tensor[c] = gaussian
        
        return blob_tensor

    def get_batch(self, idx):

        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force = item['force']
        angle = item['angle']
        

        if self.media_type == "image":
            file_path = os.path.join(self.base_path, "images", file_name)

            image = Image.open(file_path)
            desired_size = (self.width, self.height)
            if image.size != desired_size:
                print("Resizing image.")
                image = image.resize(desired_size, resample=Image.Resampling.LANCZOS)
            pixel_values = self.to_tensor_transform(image)
            pixel_values = 2*pixel_values - 1

            file_id = file_name.split(".png")[0]
            x_pos = item["coordx"] / item["width"]
            y_pos = item["coordy"] / item["height"]

        elif self.media_type == "video":

            file_path = os.path.join(self.base_path, file_name)

            pixel_values = load_video_to_pil(file_path) # [<PIL.Image.Image image mode=RGB size=832x480 at 0x7F8E842DC050>, ...], len=121 (num frames)
            pixel_values = pixel_values[:self.num_frames] # len = 81
            pixel_values = torch.stack([self.to_tensor_transform(image) for image in pixel_values])  # (81, 3, 480, 832) of torch.float32 in [0, 1]
            pixel_values = 2*pixel_values - 1 # (81, 3, 480, 832)

            file_id = file_name.split(".mp4")[0]

            # AUTOMATIC RAMDOM CROPPING PROCEDURE, but only for the carnation...
            if file_id.startswith("carnation"):

                # The video is 832x480.
                original_height, original_width = pixel_values.shape[-2:]

                # 1. ESTABLISH A CONSISTENT COORDINATE SYSTEM
                # Your `coordx` is top-left. `coordy` is bottom-left.
                # We convert `coordy` to a top-left origin for all intermediate calculations.
                coordx = item["coordx"]
                coordy_top_left = original_height - item["coordy"]

                # 2. Determine crop dimensions based on a random zoom
                crop_zoom_amount =  np.random.uniform(1.0, 1.3)
                new_width = int(original_width / crop_zoom_amount)
                new_height = int(original_height / crop_zoom_amount)

                # 3. Find a valid top-left origin (using the top-left coordinate)
                max_x_origin = original_width - new_width
                max_y_origin = original_height - new_height
                min_x_origin = max(0, int(coordx - new_width + 50))
                max_x_origin = min(max_x_origin, int(coordx - 50))
                # Use coordy_top_left here
                min_y_origin = max(0, int(coordy_top_left - new_height + 50))
                max_y_origin = min(max_y_origin, int(coordy_top_left - 50))

                if min_x_origin >= max_x_origin or min_y_origin >= max_y_origin:
                    new_origin_x_pos = np.random.randint(0, original_width - new_width + 1)
                    new_origin_y_pos = np.random.randint(0, original_height - new_height + 1)
                else:
                    new_origin_x_pos = np.random.randint(min_x_origin, max_x_origin + 1)
                    new_origin_y_pos = np.random.randint(min_y_origin, max_y_origin + 1)

                # 4. Perform the crop (this is a top-left operation)
                pixel_values = pixel_values[
                    :, :, 
                    new_origin_y_pos : new_origin_y_pos + new_height, 
                    new_origin_x_pos : new_origin_x_pos + new_width
                ]
                
                # Resize the cropped video to the final target size
                pixel_values = transforms.functional.resize(
                    pixel_values, 
                    [self.height, self.width], 
                    antialias=True
                )
                
                # 5. Calculate new coordinates relative to the crop (using top-left coordinates)
                relative_x = coordx - new_origin_x_pos
                # Use coordy_top_left here
                relative_y = coordy_top_left - new_origin_y_pos
                
                # Scale to the final pixel space. final_y is now a top-left pixel coord.
                final_x = (relative_x / new_width) * self.width
                final_y = (relative_y / new_height) * self.height

                # 6. Normalize and convert to the required final coordinate system
                # _generate_control_video needs a bottom-left normalized coordinate.
                x_pos = final_x / self.width
                y_pos = 1.0 - (final_y / self.height)
            
            else:
                x_pos = item["coordx"] / item["width"]
                y_pos = item["coordy"] / item["height"]

        return pixel_values, caption, force, angle, x_pos, y_pos, file_id


    def __getitem__(self, data_id):

        (
            pixel_values,       # (49, 3, 480, 832)
            caption,            # str
            force,              # np.float64
            angle,              # np.float64
            x_pos,              # np.float64
            y_pos,              # np.float64
            file_id             # str
        ) = self.get_batch(data_id % len(self.df))

        control_video = self._generate_control_video(
            force, angle, x_pos, y_pos, num_frames=self.num_frames, 
            num_channels=3, height=self.height, width=self.width) 

        # convert the pixel values back to a PIL image list, for compatibility
        pixel_values = (pixel_values + 1) / 2 # make values in [0,1]
        if not self.is_validation_dataset:
            # during training, the tensor is a list of image frames
            pil_image_list = [self.to_pil_transform(tensor) for tensor in pixel_values]
        else:
            # during training, the tensor just a single image frame
            pil_image_list = [self.to_pil_transform(pixel_values)]

        data = {
            "video": pil_image_list,
            "prompt": caption,
            "control_video": control_video,
            "force": force,
            "angle": angle,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "file_id": file_id,
        }

        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.df) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True

