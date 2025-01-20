from download_data import download_abide_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import logging
from sklearn.impute import SimpleImputer
from nilearn import input_data, datasets
import os
import random
import joblib  # Import joblib for saving the model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_abide_dataset(data_dir):
    """
    Preprocesses the ABIDE dataset to extract functional connectivity features.
    """
    logging.info("Loading and preprocessing ABIDE dataset...")

    # Load the dataset from the specified directory
    logging.info(f"Fetching dataset from {data_dir}...")
    abide = datasets.fetch_abide_pcp(data_dir=data_dir, pipeline='cpac', derivatives=['func_preproc'])  # Removed n_subjects

    # Debug: Inspect the structure of phenotypic data
    logging.info(f"Type of phenotypic data: {type(abide.phenotypic)}")
    logging.info(f"Columns in phenotypic data: {abide.phenotypic.columns}")
    logging.info(f"First entry of phenotypic data:\n{abide.phenotypic.iloc[0]}")

    # Initialize NiftiMasker to preprocess the fMRI data
    masker = input_data.NiftiMasker(standardize=True, mask_strategy='epi')  # Use EPI-based masking
    X = []
    y = []

    # Process each functional file and corresponding phenotypic data
    total_files = len(abide.func_preproc)
    logging.info(f"Found {total_files} functional files to process.")

    max_features = 0  # To determine the largest feature vector

    for i, (func_file, phenotypic) in enumerate(zip(abide.func_preproc, abide.phenotypic.iloc)):
        try:
            logging.info(f"Processing file {i + 1}/{total_files}: {func_file}")
            # Debug: Inspect phenotypic data
            logging.info(f"Phenotypic data for file {func_file}:\n{phenotypic}")

            # Extract time-series data
            time_series = masker.fit_transform(func_file)
            logging.info(f"Time series shape: {time_series.shape}")

            # Compute functional connectivity matrix
            correlation_matrix = np.corrcoef(time_series.T)
            logging.info(f"Correlation matrix shape: {correlation_matrix.shape}")

            # Use the upper triangle of the correlation matrix as features
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            logging.info(f"Feature length for file {func_file}: {len(upper_triangle)}")
            X.append(upper_triangle)
            max_features = max(max_features, len(upper_triangle))  # Track maximum feature size

            # Append the diagnosis label
            if 'DX_GROUP' in phenotypic:
                y.append(phenotypic['DX_GROUP'])
            else:
                logging.warning(
                    f"Phenotypic data for file {func_file} does not contain 'DX_GROUP'. Skipping this file.")
                continue
        except Exception as e:
            logging.error(f"Error processing file {func_file}: {e}")
            continue

    # Normalize features to the same length
    logging.info(f"Normalizing feature lengths to {max_features}...")
    X = [np.pad(features, (0, max_features - len(features))) for features in X]

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y) - 1  # Convert labels to 0 and 1 (e.g., ASD vs Control)
    logging.info(f"Successfully preprocessed dataset with {len(X)} samples.")
    return X, y

def augment_data(X, y, num_samples=1000):
    """
    Generates synthetic data using linear interpolation.
    """
    logging.info(f"Generating {num_samples} synthetic samples...")
    synthetic_X, synthetic_y = [], []
    for _ in range(num_samples):
        idx1, idx2 = random.sample(range(len(X)), 2)
        alpha = np.random.rand()  # Interpolation factor
        new_sample = alpha * X[idx1] + (1 - alpha) * X[idx2]
        synthetic_X.append(new_sample)
        synthetic_y.append(y[idx1])  # Assign label of the first sample
    logging.info("Synthetic data generation complete.")
    return np.array(synthetic_X), np.array(synthetic_y)

def main():
    logging.info("Starting training script...")

    # Preprocess dataset
    data_dir = './abide'  # Use the appropriate directory
    X, y = preprocess_abide_dataset(data_dir)
    input_dim = X.shape[1]
    latent_dim = 5  # Reduced latent dimension for PCA
    logging.info(f"Dataset shape: X = {X.shape}, y = {y.shape}")

    # Split dataset into train and test sets
    logging.info("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    logging.info(f"Train set shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
    logging.info(f"Test set shape: X_test = {X_test.shape}, y_test = {y_test.shape}")

    # Data augmentation (limited to real sample size)
    synthetic_X, synthetic_y = augment_data(X_train, y_train, num_samples=len(X_train))
    X_train_aug = np.vstack((X_train, synthetic_X))
    y_train_aug = np.hstack((y_train, synthetic_y))
    logging.info(f"Augmented train set shape: X_train_aug = {X_train_aug.shape}, y_train_aug = {y_train_aug.shape}")

    # Impute missing values
    logging.info("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_train_aug = imputer.fit_transform(X_train_aug)
    X_test = imputer.transform(X_test)

    # Dimensionality reduction using PCA
    logging.info(f"Reducing to {latent_dim} components with PCA...")
    pca = PCA(n_components=latent_dim)
    X_train_encoded = pca.fit_transform(X_train_aug)
    X_test_encoded = pca.transform(X_test)
    logging.info(f"Encoded train set shape: X_train_encoded = {X_train_encoded.shape}")
    logging.info(f"Encoded test set shape: X_test_encoded = {X_test_encoded.shape}")

    # Build and train SLP classifier
    logging.info("Building and training SLP classifier...")
    slp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, activation='relu',
                        solver='adam', alpha=0.01, random_state=42, early_stopping=True)

    # Use cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(slp, X_train_encoded, y_train_aug, cv=5)
    logging.info(f"Cross-validation scores: {cv_scores}")
    logging.info(f"Mean CV Accuracy: {np.mean(cv_scores) * 100:.2f}%")

    slp.fit(X_train_encoded, y_train_aug)
    logging.info("SLP classifier training complete.")

    # Evaluate the model
    y_pred = slp.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    logging.info("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model to a file
    model_filename = 'model.sh'
    joblib.dump(slp, model_filename)
    logging.info(f"Model saved to {model_filename}")

    logging.info("Training script execution complete.")

if __name__ == "__main__":
    main()