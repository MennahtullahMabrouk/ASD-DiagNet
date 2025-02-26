import sys
import os
import nibabel as nib
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel, QStatusBar
)
from pathlib import Path
import logging
from neuro.asd_one import load_model, predict_single_file  # Import ModelOne functions

# Redirect stderr to suppress macOS IMK logs
sys.stderr = open(os.devnull, 'w')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window title and size
        self.setWindowTitle("Neuro Psychic Analysis")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("Logo_Two.png"))

        # Set background color
        self.setStyleSheet("background-color: #2E3440;")

        # Central Widget Layout
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # Load Data Button
        self.create_load_data_button(central_layout)

        # Model Buttons
        self.create_model_buttons(central_layout)

        # Status Bar
        self.create_status_bar()

        # Initialize attributes
        self.file_path = None
        self.model = None
        self.pca = None
        self.imputer = None
        self.masker = None

    def create_load_data_button(self, layout):
        # Load Data Button
        load_button = QPushButton("Load Data")
        load_button.setStyleSheet(self.get_button_style())
        load_button.clicked.connect(self.load_data)
        layout.addWidget(load_button)

    def create_model_buttons(self, layout):
        # Model One Button
        model_one_button = QPushButton("Model One")
        model_one_button.setStyleSheet(self.get_button_style())
        model_one_button.clicked.connect(self.perform_model_one)
        layout.addWidget(model_one_button)

        # Model Two Button
        model_two_button = QPushButton("Model Two")
        model_two_button.setStyleSheet(self.get_button_style())
        model_two_button.clicked.connect(self.perform_model_two)
        layout.addWidget(model_two_button)

    def create_status_bar(self):
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def get_button_style(self):
        return """
            QPushButton {
                background-color: #4C566A;
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-size: 16px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
        """

    def load_data(self):
        # Open File Dialog to Browse Files
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select fMRI Data File",
            "",  # Start in the current directory
            "NIfTI Files (*.nii.gz);;1D Files (*.1D);;All Files (*)"
        )

        if file_name:
            self.file_path = Path(file_name)
            try:
                if file_name.endswith(".nii.gz"):
                    img = nib.load(file_name)
                    data = img.get_fdata()
                    print(f"Loaded .nii.gz file with shape: {data.shape}")
                    self.status_bar.showMessage("Status: .nii.gz file loaded successfully")
                elif file_name.endswith(".1D"):
                    data = np.loadtxt(file_name)
                    print(f"Loaded .1D file with shape: {data.shape}")
                    self.status_bar.showMessage("Status: .1D file loaded successfully")
                else:
                    self.status_bar.showMessage("Error: Unsupported file format.")
            except Exception as e:
                self.status_bar.showMessage(f"Error: {e}")

    def perform_model_one(self):
        if self.file_path is None:
            self.status_bar.showMessage("Error: No file loaded. Please load a file first.")
            return

        # Load the model and preprocessing objects
        model_path = Path("../ModelOne/model.joblib")  # Adjust the path as needed
        if not model_path.exists():
            self.status_bar.showMessage("Error: Model file not found. Please check the path.")
            return

        try:
            self.model, self.pca, self.imputer, self.masker = load_model(model_path)
            logging.info("Model and preprocessing objects loaded successfully.")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading model: {e}")
            return

        # Make a prediction
        try:
            result = predict_single_file(self.file_path, self.model, self.pca, self.imputer, self.masker)
            self.status_bar.showMessage(f"Model One Prediction: {result}")
            print(result)
        except Exception as e:
            self.status_bar.showMessage(f"Error during prediction: {e}")

    def perform_model_two(self):
        # Placeholder for Model Two functionality
        self.status_bar.showMessage("Model Two: Not implemented yet")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Global Style (Refined Color Scheme)
    app.setStyleSheet("""
        QMainWindow { background-color: #2E3440; }
        QLabel { font-size: 18px; font-family: Arial, sans-serif; color: white; }
        QStatusBar { background-color: #3B4252; color: white; font-size: 16px; }
    """)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())