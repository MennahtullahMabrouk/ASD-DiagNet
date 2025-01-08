import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers, models
import random
from nilearn import datasets, image, input_data

# Load and preprocess ABIDE dataset
def load_abide_dataset():
    # Load ABIDE dataset
    abide = datasets.fetch_abide_pcp(data_dir='./abide', pipeline='cpac', derivatives=['func_preproc'], n_subjects=100)

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

# Function to generate synthetic data using linear interpolation
def augment_data(X, y, num_samples=1000):
    synthetic_X, synthetic_y = [], []
    for _ in range(num_samples):
        idx1, idx2 = random.sample(range(len(X)), 2)
        alpha = np.random.rand()  # Interpolation factor
        new_sample = alpha * X[idx1] + (1 - alpha) * X[idx2]
        synthetic_X.append(new_sample)
        synthetic_y.append(y[idx1])  # Assign label of the first sample
    return np.array(synthetic_X), np.array(synthetic_y)

# Build the autoencoder model
def build_autoencoder(input_dim, latent_dim):
    input_layer = layers.Input(shape=(input_dim,))
    # Encoder
    encoded = layers.Dense(128, activation="relu")(input_layer)
    encoded = layers.Dense(latent_dim, activation="relu")(encoded)
    # Decoder
    decoded = layers.Dense(128, activation="relu")(encoded)
    decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)
    # Autoencoder model
    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    # Encoder for feature extraction
    encoder = models.Model(inputs=input_layer, outputs=encoded)
    return autoencoder, encoder

# Build the SLP classifier
def build_slp(latent_dim):
    model = models.Sequential([
        layers.Dense(32, activation="relu", input_dim=latent_dim),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# Main function to train and evaluate the ASD-DiagNet model
def main():
    # Load dataset
    X, y = load_abide_dataset()
    input_dim = X.shape[1]
    latent_dim = 64

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data augmentation
    synthetic_X, synthetic_y = augment_data(X_train, y_train)
    X_train_aug = np.vstack((X_train, synthetic_X))
    y_train_aug = np.hstack((y_train, synthetic_y))

    # Build and train autoencoder
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_train_aug, X_train_aug, epochs=50, batch_size=32, verbose=1)

    # Extract features using the trained encoder
    X_train_encoded = encoder.predict(X_train_aug)
    X_test_encoded = encoder.predict(X_test)

    # Build and train SLP classifier
    slp = build_slp(latent_dim)
    slp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    slp.fit(X_train_encoded, y_train_aug, epochs=50, batch_size=32, verbose=1)

    # Evaluate the model
    y_pred = slp.predict(X_test_encoded)
    y_pred = (y_pred > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()