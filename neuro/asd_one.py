import logging
from pathlib import Path
import os
from ModelOne.modelone import load_model, predict_single_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    model_path = Path('../ModelOne/model.joblib')

    logging.info(f"Looking for model file at: {model_path.absolute()}")

    if not model_path.exists():
        logging.error(f"Model file not found at {model_path}. Please train the model first.")
    else:
        model, pca, imputer, masker = load_model(model_path)
        logging.info("Model and preprocessing objects loaded successfully.")

        nii_file_path = Path('/Users/mennahtullahmabrouk/PycharmProjects/ASD/ASD-DiagNet/ModelOne/abide/ABIDE_pcp/cpac/nofilt_noglobal/Caltech_0051463_func_preproc.nii.gz')

        #nii_file_path = Path('ModelOne/abide/ABIDE_pcp/cpac/nofilt_noglobal/Caltech_0051463_func_preproc.nii.gz')
        logging.info(f"Looking for NIfTI file at: {nii_file_path.absolute()}")

        logging.info(f"File exists: {os.path.exists(nii_file_path)}")

        if not nii_file_path.exists():
            logging.error(f"NIfTI file not found at {nii_file_path}.")
        else:
            result = predict_single_file(nii_file_path, model, pca, imputer, masker)
            print(result)