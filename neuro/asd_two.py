import logging
from pathlib import Path
from ModelTwo.support import load_trained_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    model_path = Path("ModelTwo/model.pth")
    input_file_path = Path("ModelTwo/data/ABIDE/Outputs/cpac/filt_noglobal/rois_aal/Caltech_0051456_rois_aal.1D")  # Update this

    model = load_trained_model(model_path)
    if model:
        diagnosis, confidence = predict(input_file_path, model)
        if diagnosis:
            print(f"Predicted Diagnosis: {diagnosis} (Confidence: {confidence:.1f}%)")
