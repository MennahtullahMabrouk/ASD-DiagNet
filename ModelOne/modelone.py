import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import logging
from sklearn.impute import SimpleImputer
from nilearn import input_data, datasets
import random
import joblib
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_abide_dataset(data_dir):
    """
    Preprocesses the ABIDE dataset to extract functional connectivity features.
    """
    logging.info("Loading and preprocessing ABIDE dataset...")

    # Load the dataset from the specified directory
    logging.info(f"Fetching dataset from {data_dir}...")
    abide = datasets.fetch_abide_pcp(data_dir=data_dir, pipeline='cpac', derivatives=['func_preproc'])

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
    return X, y, masker

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

def save_model(model, pca, imputer, masker, filename='model.joblib'):
    """
    Save the trained model and preprocessing objects to a file.
    """
    logging.info(f"Saving model and preprocessing objects to {filename}...")
    joblib.dump({
        'model': model,
        'pca': pca,
        'imputer': imputer,
        'masker': masker
    }, filename)
    logging.info("Model and preprocessing objects saved.")

def load_model(filename='model.joblib'):
    """
    Load the trained model and preprocessing objects from a file.
    """
    logging.info(f"Loading model and preprocessing objects from {filename}...")
    data = joblib.load(filename)
    return data['model'], data['pca'], data['imputer'], data['masker']

def preprocess_single_file(nii_file_path, masker, imputer, pca):
    """
    Preprocess a single .nii.gz file for prediction.
    """
    logging.info(f"Preprocessing single file: {nii_file_path}")
    try:
        # Debug: Check if the file exists
        if not Path(nii_file_path).exists():
            logging.error(f"File not found at {nii_file_path}.")
            return None

        # Debug: Load the file with nibabel to verify it can be read
        import nibabel as nib
        try:
            img = nib.load(nii_file_path)
            logging.info(f"NIfTI file loaded successfully. Shape: {img.shape}")
        except Exception as e:
            logging.error(f"Error loading NIfTI file with nibabel: {e}")
            return None

        # Load the .nii.gz file with NiftiMasker
        logging.info("Loading file with NiftiMasker...")
        time_series = masker.transform(nii_file_path)
        logging.info(f"Time series shape: {time_series.shape}")

        # Compute functional connectivity matrix
        logging.info("Computing correlation matrix...")
        correlation_matrix = np.corrcoef(time_series.T)
        logging.info(f"Correlation matrix shape: {correlation_matrix.shape}")

        # Use the upper triangle of the correlation matrix as features
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        logging.info(f"Feature length: {len(upper_triangle)}")

        # Normalize feature length
        max_features = pca.n_features_in_
        features = np.pad(upper_triangle, (0, max_features - len(upper_triangle)))

        # Impute missing values and apply PCA
        logging.info("Imputing missing values and applying PCA...")
        features = imputer.transform([features])
        features_encoded = pca.transform(features)
        return features_encoded
    except Exception as e:
        logging.error(f"Error preprocessing file {nii_file_path}: {e}")
        return None

def save_predictions(nii_file_path, features, prediction, prediction_proba, output_file='predictions.csv'):
    """
    Save predictions and input features to a CSV file.
    """
    try:
        # Create a DataFrame with the prediction results
        data = {
            'file_path': [nii_file_path],
            'features': [features.tolist()],  # Save features as a list
            'prediction': [prediction[0]],  # Save the predicted class
            'confidence_asd': [prediction_proba[1]],  # Confidence for ASD
            'confidence_no_asd': [prediction_proba[0]]  # Confidence for no ASD
        }
        df = pd.DataFrame(data)

        # Append to the existing file or create a new one
        if Path(output_file).exists():
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, index=False)

        logging.info(f"Prediction saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving prediction: {e}")

def predict_single_file(nii_file_path, model, pca, imputer, masker, save_to_file=True):
    """
    Preprocess a single .nii.gz file and make a prediction.
    Optionally save the prediction to a file.
    """
    logging.info(f"Predicting for single file: {nii_file_path}")
    try:
        # Check if the file exists
        if not Path(nii_file_path).exists():
            logging.error(f"File not found at {nii_file_path}.")
            return "Error: File not found."

        # Preprocess the file
        logging.info("Preprocessing the file...")
        preprocessed_data = preprocess_single_file(nii_file_path, masker, imputer, pca)
        if preprocessed_data is None:
            return "Error: Unable to preprocess the file."

        # Make a prediction
        logging.info("Making prediction...")
        prediction = model.predict(preprocessed_data)
        prediction_proba = model.predict_proba(preprocessed_data)[0]  # Get prediction probabilities

        # Save the prediction to a file
        if save_to_file:
            save_predictions(nii_file_path, preprocessed_data, prediction, prediction_proba)

        # Map prediction to human-readable message
        if prediction[0] == 1:
            return f"Patient likely has ASD (Confidence: {prediction_proba[1] * 100:.2f}%)"
        else:
            return f"Patient does not have ASD (Confidence: {prediction_proba[0] * 100:.2f}%)"
    except Exception as e:
        logging.error(f"Error predicting for file {nii_file_path}: {e}")
        return "Error: Unable to make a prediction."

def load_predictions(prediction_file='predictions.csv'):
    """
    Load saved predictions from a file.
    """
    try:
        df = pd.read_csv(prediction_file)
        logging.info(f"Loaded {len(df)} predictions from {prediction_file}")
        return df
    except Exception as e:
        logging.error(f"Error loading predictions: {e}")
        return None

def retrain_model_with_predictions(X_train, y_train, prediction_file='predictions.csv'):
    """
    Retrain the model using the original dataset and saved predictions.
    """
    # Load saved predictions
    predictions_df = load_predictions(prediction_file)
    if predictions_df is None:
        logging.warning("No predictions found. Retraining with original dataset only.")
        return X_train, y_train

    # Extract features and labels from predictions
    X_new = np.array([eval(features) for features in predictions_df['features']])  # Convert string to list
    y_new = predictions_df['prediction'].values

    # Combine with the original dataset
    X_train_combined = np.vstack((X_train, X_new))
    y_train_combined = np.hstack((y_train, y_new))

    logging.info(f"Combined dataset shape: X_train = {X_train_combined.shape}, y_train = {y_train_combined.shape}")
    return X_train_combined, y_train_combined

def main():
    logging.info("Starting training script...")

    # Preprocess dataset
    data_dir = './abide'  # Use the appropriate directory
    X, y, masker = preprocess_abide_dataset(data_dir)
    input_dim = X.shape[1]
    latent_dim = 5  # Reduced latent dimension for PCA
    logging.info(f"Dataset shape: X = {X.shape}, y = {y.shape}")

    # Split dataset into train and test sets
    logging.info("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    logging.info(f"Train set shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
    logging.info(f"Test set shape: X_test = {X_test.shape}, y_test = {y_test.shape}")

    # Retrain the model with saved predictions
    X_train_combined, y_train_combined = retrain_model_with_predictions(X_train, y_train)

    # Data augmentation (limited to real sample size)
    synthetic_X, synthetic_y = augment_data(X_train_combined, y_train_combined, num_samples=len(X_train_combined))
    X_train_aug = np.vstack((X_train_combined, synthetic_X))
    y_train_aug = np.hstack((y_train_combined, synthetic_y))
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

    # Save the trained model and preprocessing objects
    save_model(slp, pca, imputer, masker, filename='model.joblib')

    # Example: Predict for a single file
    nii_file_path = 'path/to/your/file.nii.gz'  # Replace with the actual file path
    result = predict_single_file(nii_file_path, slp, pca, imputer, masker)
    print(result)

    logging.info("Training script execution complete.")

if __name__ == "__main__":
    main()