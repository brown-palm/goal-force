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

import requests
import os

# List of (filename, download URL, save_dir)
files_to_download = [
    (
        "step-3000.safetensors",
        "https://huggingface.co/brown-palm/goal-force/resolve/main/step-3000.safetensors",
        "checkpoints/goal_force",
    ),
    (
        "step-500.safetensors",
        "https://huggingface.co/brown-palm/wan2.2_controlnet_canny_edge/resolve/main/step-500.safetensors",
        "checkpoints/wan2.2_controlnet_canny_edge",
    ),
]

# Download each file
for filename, url, save_dir in files_to_download:

    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(save_dir, filename), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {os.path.join(save_dir, filename)}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")
