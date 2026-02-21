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
Script to extract the first frame from a video.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffsynth import VideoData


def extract_first_frame(input_video_path: str, output_image_path: str):
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading video from: {input_video_path}")
    
    input_image = VideoData(input_video_path, height=480, width=832)[0]
    input_image.save(output_image_path)
    
    print(f"Successfully saved first frame to: {output_image_path}")


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
        help='Path where the output image will be saved (e.g., /path/to/video_first_frame.png)'
    )
    
    args = parser.parse_args()
    
    try:
        extract_first_frame(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

