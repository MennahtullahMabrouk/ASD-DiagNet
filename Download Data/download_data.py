import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_abide_dataset(folder_path, phenotypic_file="./abide/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"):
    """
    Load ABIDE dataset from the folder and phenotypic file.
    Handles missing or unmatched functional files and labels gracefully.
    """
    print("Loading dataset directly from folder...")

    # Load phenotypic file
    phenotypic_data = pd.read_csv(phenotypic_file)
    file_to_label = {row["FILE_ID"]: row["DX_GROUP"] for _, row in phenotypic_data.iterrows() if
                     row["FILE_ID"] != "no_filename"}

    print(f"Labels available in phenotypic file: {list(file_to_label.keys())[:10]} (showing first 10)")

    # Get all functional files in the folder
    func_files = [f for f in os.listdir(folder_path) if f.endswith("_func_preproc.nii.gz")]
    print(f"Files found in the folder: {func_files[:10]} (showing first 10)")

    X, y = [], []
    unmatched_files = []
    for func_file in tqdm(func_files, desc="Processing files"):
        file_id = func_file.split("_func_preproc")[0]
        if file_id in file_to_label:
            # Add the label and placeholder data (replace with actual .nii.gz data loading logic)
            y.append(file_to_label[file_id])
            X.append(np.random.rand(100, 100))  # Example placeholder for functional data
        else:
            unmatched_files.append(func_file)
            print(f"Skipping unmatched file: {func_file}")

    if not X or not y:
        raise ValueError(
            f"No valid functional files or labels found. Unmatched files: {unmatched_files[:10]} (showing first 10). "
            "Ensure FILE_IDs in the phenotypic file match filenames in the folder."
        )

    print(f"Dataset loading completed. Total valid samples: {len(X)}")
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Test download and save dataset
    X, y = load_abide_dataset()
    print(f"Downloaded data shape: X={X.shape}, y={y.shape}")
