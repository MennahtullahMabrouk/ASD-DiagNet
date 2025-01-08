import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

# Define the S3 bucket and prefix for the ABIDE dataset
S3_BUCKET = "fcp-indi"
S3_PREFIX = "data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/"

# Define the local directory to save the data
SAVE_DIR = "./abide_data"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_s3_directory(bucket_name, prefix, local_dir):
    """
    Downloads all files from an S3 directory to a local directory.
    Skips files that already exist locally.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))  # Use unsigned requests
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    # Get the list of files to download
    files_to_download = []
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                files_to_download.append(obj["Key"])

    if not files_to_download:
        print(f"No files found in S3 bucket: {bucket_name}/{prefix}")
        return

    # Download each file (skip if already exists)
    for file_key in tqdm(files_to_download, desc="Downloading files"):
        local_file_path = os.path.join(local_dir, os.path.relpath(file_key, prefix))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Skip download if the file already exists
        if os.path.exists(local_file_path):
            print(f"File already exists: {local_file_path}. Skipping download.")
            continue

        # Download the file
        s3.download_file(bucket_name, file_key, local_file_path)

def download_abide_data():
    """
    Downloads the ABIDE dataset (phenotypic and functional data).
    Skips download if files already exist locally.
    """
    print("Downloading ABIDE dataset...")

    # Download phenotypic data
    phenotypic_save_path = os.path.join(SAVE_DIR, "Phenotypic_V1_0b.csv")
    if not os.path.exists(phenotypic_save_path):
        print(f"Downloading phenotypic data to {phenotypic_save_path}...")
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        s3.download_file(
            S3_BUCKET,
            "data/Projects/ABIDE_Initiative/Phenotypic_V1_0b.csv",
            phenotypic_save_path,
        )
    else:
        print(f"Phenotypic data already exists at {phenotypic_save_path}.")

    # Download functional data (C-PAC pipeline)
    functional_save_dir = os.path.join(SAVE_DIR, "functionals", "cpac", "filt_global", "rois_cc200")
    os.makedirs(functional_save_dir, exist_ok=True)

    print(f"Downloading functional data to {functional_save_dir}...")
    download_s3_directory(S3_BUCKET, S3_PREFIX, functional_save_dir)

    print("Download complete!")

if __name__ == "__main__":
    download_abide_data()