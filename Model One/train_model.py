import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import os
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

# Function to generate synthetic data using SMOTE
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

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
        layers.Dense(64, activation="relu", input_dim=latent_dim),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# Main function to train and evaluate the ASD-DiagNet model
def main():
    # Load dataset
    data = np.load('abide_dataset.npz')
    X, y = data['X'], data['y']
    input_dim = X.shape[1]
    latent_dim = 64

    # Balance the dataset using SMOTE
    X_balanced, y_balanced = balance_data(X, y)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Build and train autoencoder
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

    # Save the autoencoder and encoder
    autoencoder.save('autoencoder_model.h5')
    encoder.save('encoder_model.h5')
    print("Autoencoder and encoder saved.")

    # Extract features using the trained encoder
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # Build and train SLP classifier
    slp = build_slp(latent_dim)
    slp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    slp.fit(X_train_encoded, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

    # Save the SLP classifier
    slp.save('slp_classifier.h5')
    print("SLP classifier saved.")

    # Save the full model as asdmodelone.h5
    full_model = models.Sequential([encoder, slp])
    full_model.save('asdmodelone.h5')
    print("Full model saved as asdmodelone.h5.")

    # Generate and save the model architecture image
    os.makedirs('architecture', exist_ok=True)  # Create the architecture directory
    plot_model(full_model, to_file='architecture/modelonearchitecture.png', show_shapes=True, show_layer_names=True)
    print("Model architecture image saved as architecture/modelonearchitecture.png.")

    # Evaluate the model
    y_pred = slp.predict(X_test_encoded)
    y_pred = (y_pred > 0.5).astype(int)

    # Print evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, y_pred))

if __name__ == "__main__":
    main()