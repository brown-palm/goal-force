"""
MIT License

Copyright (c) 2025 Nate Gillman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Script to extract Canny edges from a video.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from controlnet_aux import CannyDetector
from diffsynth import save_video

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.goal_force.unified_dataset import ControlSignalDataset_CannyEdge


def extract_canny_edges(input_video_path: str, output_video_path: str, num_frames: int):
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading video from: {input_video_path}")
    print(f"Using number of frames: {num_frames}")
    
    video_operator = ControlSignalDataset_CannyEdge.default_video_operator(
        base_path="",
        max_pixels=921600,
        height=480,
        width=832,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=num_frames,
        time_division_factor=4,
        time_division_remainder=1,
    )
    
    processed_video = video_operator(input_video_path)
    
    if processed_video is None:
        raise ValueError(f"Failed to load video from: {input_video_path}")
    
    canny_detector = CannyDetector()
    
    video_as_np = np.array(processed_video)  # (num_frames, 480, 832, 3), uint8's in [0, 255]
    
    canny_frames = []
    for frame in video_as_np:
        canny = canny_detector(frame)  # (512, 896, 3)
        if canny.shape[:2] != frame.shape[:2]:
            canny = cv2.resize(canny, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        
        canny_frames.append(canny)
    
    canny_frames_np = np.stack(canny_frames, axis=0)  # (num_frames, 480, 832, 3)
    print(f"Canny edge video shape: {canny_frames_np.shape}")
    
    save_video(canny_frames_np, output_video_path, fps=15, quality=5)
    print(f"Successfully saved Canny edge video to: {output_video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input video file (e.g., /path/to/video.mp4)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path where the output Canny edge video will be saved (e.g., /path/to/video_canny.mp4)'
    )
    parser.add_argument(
        '--num_frames', '-n',
        type=int,
        default=81,
        help='Number of frames to extract'
    )
    
    args = parser.parse_args()
    
    try:
        extract_canny_edges(args.input, args.output, args.num_frames)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

