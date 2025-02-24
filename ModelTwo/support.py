import argparse
import logging
from pathlib import Path
import joblib
import torch
from support import MTAutoEncoder, preprocess_input, predict
from modeltwo import train_model  # Import training function

# ======== Set Up Logging ========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ModelTwo/model.joblib"  # Path to trained model

# ======== Load Trained Model ========
def load_trained_model():
    """
    Load the trained model from a joblib file.
    """
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        return None

    model_state_dict = joblib.load(MODEL_PATH)  # Load model state dict
    num_inputs = 116  # Ensure it matches training
    num_latent = num_inputs // 2

    # Reinitialize the model and load state dict
    model = MTAutoEncoder(num_inputs=num_inputs, num_latent=num_latent, tied=True, use_dropout=False)
    model.load_state_dict(model_state_dict)
    model.eval()  # Set to evaluation mode
    logger.info("Trained model loaded successfully.")
    return model

# ======== Main Execution ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Predict using MTAutoEncoder")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, nargs="+", help="Path(s) to .1D file(s) for prediction")

    args = parser.parse_args()

    if args.train:
        train_model()  # Run training

    elif args.predict:
        model = load_trained_model()
        if model:
            filepaths = [Path(f) for f in args.predict if Path(f).exists()]
            if not filepaths:
                logger.error("No valid input files found.")
            else:
                results = predict(filepaths, model)
                for filepath, (diagnosis, confidence) in results.items():
                    if diagnosis:
                        print(f"{filepath}: {diagnosis} (Confidence: {confidence:.1f}%)")

    else:
        logger.error("Please specify either --train to train the model or --predict <file_path> to predict.")
