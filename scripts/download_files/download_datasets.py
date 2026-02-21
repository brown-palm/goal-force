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

import os
import shutil
import zipfile
from huggingface_hub import snapshot_download

# datasets/
# ├── train/
# │   └── balls_6k/
# │   └── dominos_3k/
# │   └── plants_3k/
# │   └── balls_6k.csv
# │   └── dominos_3k.csv
# │   └── plants_3k.csv


# === Configuration ===
repo_id = "brown-palm/goal-force-training-datasets"
download_root = "datasets"  # Top-level output directory
local_dir="hf_temp_datasets"

# === Step 1: Download entire repo snapshot ===
repo_snapshot_path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

# === Step 2: Define paths ===
zip_path = os.path.join(repo_snapshot_path, "dataset_videos.zip")
csv1_src = os.path.join(repo_snapshot_path, "balls_6k.csv")
csv2_src = os.path.join(repo_snapshot_path, "dominos_3k.csv")
csv3_src = os.path.join(repo_snapshot_path, "plants_3k.csv")

# === Step 3: Extract ZIP to a temp location ===
temp_extract_dir = os.path.join(download_root, "_extracted_temp")
os.makedirs(temp_extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(temp_extract_dir)

# === Step 4: Move folders and CSVs into nested structure ===
# Define target paths
target1_dir = os.path.join(download_root, "train", "balls_6k")
target1_csv = os.path.join(download_root, "train", "balls_6k.csv")
target2_dir = os.path.join(download_root, "train", "dominos_3k")
target2_csv = os.path.join(download_root, "train", "dominos_3k.csv")
target3_dir = os.path.join(download_root, "train", "plants_3k")
target3_csv = os.path.join(download_root, "train", "plants_3k.csv")

# Move video folders
shutil.move(os.path.join(temp_extract_dir, "balls_6k"), target1_dir)
shutil.move(os.path.join(temp_extract_dir, "dominos_3k"), target2_dir)
shutil.move(os.path.join(temp_extract_dir, "plants_3k"), target3_dir)

# Move CSVs
os.makedirs(os.path.dirname(target1_csv), exist_ok=True)
os.makedirs(os.path.dirname(target2_csv), exist_ok=True)
shutil.copy(csv1_src, target1_csv)
shutil.copy(csv2_src, target2_csv)
shutil.copy(csv3_src, target3_csv)

# Clean up temporary extract folder
shutil.rmtree(temp_extract_dir)

# Clean up temporary local dir
shutil.rmtree(local_dir)

print("✅ Downloaded and organized dataset successfully.")
