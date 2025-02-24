## ASD-DiagNet: A hybrid learning approach for detection of Autism Spectrum Disorder using fMRI data
https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2019.00070/full

### Dependencies
```
Python 3.9 
Numpy == 2.0.2
Nilearn == 0.11.1
Tensorflow == 2.18.0
Scikit-Learn == 1.6.0
```

### Dataset
http://preprocessed-connectomes-project.org/abide/

## **Model 1: Autoencoder + SLP (TensorFlow/Keras)**

### **Technical Details**:
1. **Autoencoder Architecture**:
   - **Encoder**: Two dense layers with ReLU activation.
     - Input dimension → 128 units → Latent dimension.
   - **Decoder**: Two dense layers with ReLU and sigmoid activation.
     - Latent dimension → 128 units → Input dimension.
   - The autoencoder is trained to minimize the **Mean Squared Error (MSE)** between the input and reconstructed output.

2. **Classifier Architecture**:
   - **Single-Layer Perceptron (SLP)**:
     - Input: Latent features from the encoder.
     - Architecture: Dense layer (64 units, ReLU) → Dropout (0.5) → Output layer (1 unit, sigmoid).
   - The classifier is trained using **binary cross-entropy loss**.

3. **Data Preprocessing**:
   - **SMOTE**: Generates synthetic samples for the minority class to balance the dataset.
   - **Normalization**: Input data is normalized (if required) before feeding into the autoencoder.

4. **Training Process**:
   - The autoencoder is trained first to learn meaningful latent representations.
   - The SLP is then trained on the latent features extracted by the encoder.

5. **Evaluation**:
   - Metrics: Accuracy, classification report (precision, recall, F1-score), and AUC-ROC score.
   - The model is evaluated on a **hold-out test set** (20% of the data).

6. **Saving and Visualization**:
   - The autoencoder, encoder, and SLP are saved as `.h5` files.
   - The model architecture is visualized using `plot_model` and saved as a PNG image.

---

### **Performance Considerations**:
1. **Strengths**:
   - **Ease of Use**: High-level Keras APIs make it easy to implement and experiment.
   - **Class Imbalance Handling**: SMOTE effectively addresses class imbalance.
   - **Visualization**: Provides a clear visual representation of the model architecture.

2. **Weaknesses**:
   - **Limited Generalization**: The use of a simple train/test split may lead to overfitting.
   - **No Domain-Specific Augmentation**: SMOTE is generic and does not leverage domain knowledge (e.g., fMRI data characteristics).

---

### **Practical Implications**:
- **Deployment**: The model is easy to deploy due to its simplicity and compatibility with TensorFlow Serving or other deployment tools.
- **Scalability**: Suitable for small to medium-sized datasets but may struggle with very large datasets due to the lack of advanced optimization techniques.

---

## **Model 2: ASD-DiagNet (PyTorch)**

### **Technical Details**:
1. **Autoencoder Architecture**:
   - **Encoder**: One dense layer with tanh activation.
     - Input dimension → Latent dimension.
   - **Decoder**: One dense layer with tanh activation (if not tied) or uses the encoder’s weights (if tied).
     - Latent dimension → Input dimension.
   - The autoencoder is trained to minimize the **Mean Squared Error (MSE)**.

2. **Classifier Architecture**:
   - **Feedforward Neural Network**:
     - Input: Latent features from the encoder.
     - Architecture: Dense layer (latent dimension → 1 unit, sigmoid).
   - The classifier is trained using **binary cross-entropy loss**.

3. **Data Preprocessing**:
   - **Correlation Matrices**: Computes correlation matrices from fMRI data.
   - **Eigenvector-Based Augmentation**: Generates synthetic samples using eigenvector similarity.

4. **Training Process**:
   - The autoencoder and classifier are trained jointly using a combined loss function:
     - Reconstruction loss (MSE) for the autoencoder.
     - Classification loss (binary cross-entropy) for the classifier.

5. **Evaluation**:
   - Metrics: Accuracy, sensitivity, and specificity.
   - The model is evaluated using **10-fold cross-validation**, providing a more robust estimate of performance.

6. **Saving and Visualization**:
   - The model state dictionary is saved as a `.sh` file.
   - The model architecture is visualized using `torchviz` and saved as a PNG image.

---

### **Performance Considerations**:
1. **Strengths**:
   - **Robust Evaluation**: 10-fold cross-validation provides a more reliable estimate of model performance.
   - **Domain-Specific Augmentation**: Eigenvector-based augmentation leverages domain knowledge to improve generalization.
   - **Flexibility**: PyTorch allows for dynamic computation graphs and custom implementations.

2. **Weaknesses**:
   - **Complexity**: The implementation is more complex due to PyTorch and custom augmentation.
   - **Training Time**: The use of cross-validation and augmentation increases training time.

---

### **Practical Implications**:
- **Deployment**: The model can be deployed using PyTorch’s `torchscript` or ONNX for compatibility with production systems.
- **Scalability**: Suitable for both small and large datasets due to PyTorch’s efficient computation and support for distributed training.

---

## **Detailed Comparison**

| Feature                        | **Model 1 (Autoencoder + SLP)**          | **Model 2 (ASD-DiagNet)**                  |
|--------------------------------|------------------------------------------|--------------------------------------------|
| **Framework**                  | TensorFlow/Keras                        | PyTorch                                    |
| **Input Data**                 | Raw fMRI data (flattened)               | Correlation matrices                       |
| **Preprocessing**              | SMOTE for class imbalance               | Eigenvector-based augmentation             |
| **Autoencoder Architecture**   | Encoder: 128 → Latent, Decoder: Latent → 128 → Input | Encoder: Input → Latent, Decoder: Latent → Input |
| **Classifier Architecture**    | Single-Layer Perceptron (SLP)           | Feedforward neural network                 |
| **Training Process**           | Autoencoder → SLP                       | Joint training of autoencoder and classifier |
| **Evaluation**                 | Train/test split                        | 10-fold cross-validation                   |
| **Metrics**                    | Accuracy, Classification Report, AUC-ROC| Accuracy, Sensitivity, Specificity         |
| **Visualization**              | Yes (model architecture as PNG)         | Yes (model architecture as PNG)            |
| **Complexity**                 | Low (high-level APIs)                   | High (custom implementation)               |
| **Use Case**                   | Quick prototyping, class imbalance      | Research, advanced applications            |

---

