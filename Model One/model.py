import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Function to load dataset
def load_abide_dataset():
    """Simulates dataset loading. Replace with actual dataset loading logic."""
    # Example: 100 samples, 100 features, binary classification labels
    X = np.random.rand(100, 100)
    y = np.random.choice([0, 1], size=(100,))  # Binary classification labels
    return X, y

# Function to augment data
def augment_data(X, y):
    """Simulates data augmentation. Replace with actual augmentation logic."""
    synthetic_X = X + np.random.normal(0, 0.01, X.shape)
    synthetic_y = y.copy()  # Simple copy; modify logic for meaningful augmentation
    return synthetic_X, synthetic_y

# Function to build autoencoder
def build_autoencoder(input_dim, latent_dim):
    """Builds an autoencoder model."""
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(latent_dim, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    return autoencoder, encoder

# Function to build classifier
def build_slp(input_dim):
    """Builds a simple feed-forward neural network for classification."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def main():
    # Load dataset
    print("Loading dataset...")
    X, y = load_abide_dataset()
    input_dim = X.shape[1]
    latent_dim = 64

    # Split dataset into train and test sets
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data augmentation
    print("Augmenting data...")
    synthetic_X, synthetic_y = augment_data(X_train, y_train)
    X_train_aug = np.vstack((X_train, synthetic_X))
    y_train_aug = np.hstack((y_train, synthetic_y))

    # Build and train autoencoder
    print("Building and training autoencoder...")
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_train_aug, X_train_aug, epochs=50, batch_size=32, verbose=1)

    # Extract features using the trained encoder
    print("Encoding features...")
    X_train_encoded = encoder.predict(X_train_aug)
    X_test_encoded = encoder.predict(X_test)

    # Build and train SLP classifier
    print("Building and training classifier...")
    slp = build_slp(latent_dim)
    slp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    slp.fit(X_train_encoded, y_train_aug, epochs=50, batch_size=32, verbose=1)

    # Evaluate the model
    print("Evaluating model...")
    y_pred = slp.predict(X_test_encoded)
    y_pred = (y_pred > 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Optional: Detailed classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
