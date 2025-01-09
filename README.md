# Early Detection of Alzheimer’s Disease Using MRI Analysis

## Project Overview
This project utilizes convolutional neural networks (CNNs) to classify MRI scans into stages of Alzheimer’s disease: **non-demented**, **very mild dementia**, **mild dementia**, and **moderate dementia**. The project focuses on improving early detection by addressing borderline cases through advanced image preprocessing and deep learning techniques.

---

## File Organization

| File                        | Purpose                                                                                  | Input                          | Output                          |
|-----------------------------|------------------------------------------------------------------------------------------|--------------------------------|---------------------------------|
| `3D-AD-Detection-CNN.ipynb` | Trains a 3D CNN for Alzheimer’s classification on OASIS, using stacked MRI slices.       | 3D MRI tensors, hyperparameters | Trained model, performance data |
| `AD-Detection-CNN.ipynb`    | Implements a 2D CNN for classification using individual slices.                          | 2D MRI slices, labels          | Accuracy, loss curves           |
| `test_on_diff_dataset.ipynb`| Evaluates generalization of the OASIS-trained model on the Kaggle dataset.               | Kaggle dataset, trained model  | Evaluation metrics              |
| `transfer_learning.ipynb`   | Fine-tunes OASIS-trained models on the Kaggle dataset.                                  | Pre-trained model, new dataset | Updated weights, metrics        |
| `2d_cnn_model.pth`          | Stores weights of the trained 2D CNN.                                                  | -                              | Model weights                   |
| `3d_cnn_model.pth`          | Stores weights of the trained 3D CNN.                                                  | -                              | Model weights                   |

---

## Results

### Key Findings
- **2D CNN**: Initial high accuracy (>99%) due to data leakage was resolved by treating 61 related slices as a 3D scan, revealing true complexity.
- **3D CNN**: Achieved strong performance in binary classification (demented vs. non-demented) with balanced datasets.
- **Transfer Learning**: Highlighted challenges due to preprocessing differences and dataset variability.

### Metrics
- **Precision, Recall, F1 Score**: Evaluated classification performance, especially for mild and very mild dementia.
- **Confusion Matrices**: Analyzed misclassification patterns to identify areas for improvement.

---

## How to Run
1. **Train Models**:  
   - For 2D: Run `AD-Detection-CNN.ipynb`.  
   - For 3D: Run `3D-AD-Detection-CNN.ipynb`.  
2. **Evaluate Performance**:  
   Use `test_on_diff_dataset.ipynb` to measure generalizability on the Kaggle dataset.
3. **Fine-Tune Models**:  
   Run `transfer_learning.ipynb` to adapt OASIS-trained models to new data.

---

## Research Paper
A detailed report on our methodology, results, and findings is available in the [`docs`](docs) directory:  
📄 [Read the Full Research Paper](docs/AlzheimerDetection_Research_Paper.pdf)

---

## Future Directions
- Automate preprocessing pipelines for compatibility with multiple datasets.  
- Optimize CNN architectures for better handling of 3D inputs.  
- Explore ensemble methods combining multiple CNN architectures for improved accuracy.  
- Address class imbalance using advanced data augmentation techniques like GANs.

---

## Acknowledgments
This project builds upon the datasets and insights from:  
- [OASIS Dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/)  
- [Enhanced Kaggle Dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/)  

Special thanks to the contributors of this repository:  
- **Ben Boben**  
- **Rohan Joseph Jayasekara**  
- **Jason Dunn**  

References:  
1. Shaffi, "Ensemble of Vision Transformer Architectures for Alzheimer’s Disease Classification."  
2. Murugan, "DEMNET: Early Diagnosis of Alzheimer’s Disease from MRI Images."  

---
