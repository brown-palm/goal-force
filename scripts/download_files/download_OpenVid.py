import os
import subprocess
import argparse


def download_files(output_directory, part_idx):

    # STEP 1: download CSVs
    data_folder = output_directory
    os.makedirs(data_folder, exist_ok=True)
    data_urls = [
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv",
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv"
    ]
    for data_url in data_urls:
        data_path = os.path.join(data_folder, os.path.basename(data_url))
        command = ["wget", "-O", data_path, data_url]
        subprocess.run(command, check=True)

    # STEP 2: download videos
    zip_folder = os.path.join(output_directory, "download")
    video_folder = os.path.join(output_directory, "video")
    os.makedirs(zip_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    # default is "for i in range(0, 186)". 
    # in the interest of getting it up and running, we only download part of it...
    if part_idx == None:
        MIN_TO_DOWNLOAD = 36 # default: 0
        MAX_TO_DOWNLOAD = 36 # default: 185
    else:
        MIN_TO_DOWNLOAD, MAX_TO_DOWNLOAD = part_idx, part_idx

    for i in range(MIN_TO_DOWNLOAD, MAX_TO_DOWNLOAD+1):
        url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}.zip"
        file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")

        download_command = ["wget", "-O", file_path, url]
        unzip_command = ["unzip", "-j", file_path, "-d", video_folder]

        try:
            subprocess.run(download_command, check=True)
            print(f"file {url} saved to {file_path}")
            subprocess.run(unzip_command, check=True)
        except subprocess.CalledProcessError as e:
            error_message = f"file {url} download failed: {e}\n"
            print(error_message)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some parameters.')

    parser.add_argument('--output_directory', type=str, help='Path to the dataset directory', default="datasets/OpenVid-1M/OpenVid-1M-train")
    parser.add_argument('--part_idx', type=int, default=None)
    args = parser.parse_args()
    download_files(args.output_directory, args.part_idx)

