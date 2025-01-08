import numpy as np
from nilearn import datasets, input_data
import os

# Load and preprocess ABIDE dataset
def load_abide_dataset():
    # Define the data directory
    data_dir = './abide'

    # Check if the dataset already exists
    if not os.path.exists(data_dir):
        print("Downloading ABIDE dataset...")
        os.makedirs(data_dir, exist_ok=True)
    else:
        print("ABIDE dataset already exists. Skipping download...")

    # Load ABIDE dataset (full dataset)
    abide = datasets.fetch_abide_pcp(data_dir=data_dir, pipeline='cpac', derivatives=['func_preproc'])

    # Extract functional connectivity features using Nilearn
    masker = input_data.NiftiMasker(standardize=True)
    X = []
    y = []

    for func_file, phenotypic in zip(abide.func_preproc, abide.phenotypic):
        try:
            time_series = masker.fit_transform(func_file)  # Extract time-series data
            correlation_matrix = np.corrcoef(time_series.T)  # Functional connectivity
            X.append(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])  # Upper triangle as features
            y.append(phenotypic['DX_GROUP'])  # Diagnosis label
        except:
            continue

    X = np.array(X)
    y = np.array(y) - 1  # Convert labels to 0 and 1 (e.g., ASD vs Control)
    return X, y

if __name__ == "__main__":
    # Save the dataset to a file
    X, y = load_abide_dataset()
    np.savez('abide_dataset.npz', X=X, y=y)
    print("Dataset saved to 'abide_dataset.npz'.")