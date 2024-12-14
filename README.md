# Alzheimer’s Disease Detection Project  

## Overview  
This project aims to detect Alzheimer’s disease through convolutional neural networks (CNNs) trained on MRI datasets. It includes scripts and frameworks for training, evaluating, and fine-tuning models across different datasets, specifically focusing on classifying MRI scans into stages of dementia.  

## Files and Descriptions  

| **File**                   | **Purpose**                                                                                  | **Input**                                                | **Output**                                           |
|----------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------|
| `3D-AD-Detection-CNN.ipynb`| Trains and evaluates a CNN for Alzheimer’s classification using the OASIS dataset, combining 61 2D images into a 3D scan for classification. | 3D MRI data slices, hyperparameters | Trained CNN model, performance data |
| `AD-Detection-CNN.ipynb`   | Trains and evaluates a CNN for Alzheimer’s classification using the OASIS dataset, processing individual 2D images. | 2D MRI images, labels | Classification accuracy, loss curves |
| `test_on_diff_dataset.ipynb`| Evaluates the Kaggle dataset using a model pre-trained on the OASIS dataset to measure generalization capability. | Kaggle dataset, model weights | Evaluation metrics |
| `transfer_learning.ipynb`  | Fine-tunes a pre-trained model on the Kaggle dataset, exploring transfer learning. | Pre-trained model, Kaggle dataset  | Updated model weights, transfer learning metrics |
| `2d_cnn_model.pth	` | Stores weights of the trained CNN for individual 2D images. | - | Model weights |
| `3d_cnn_model.pth	` | Stores weights of the trained CNN for individual 3D scan. | - | Model weights |

## Instructions  
1. Run `3D-AD-Detection-CNN.ipynb` or `AD-Detection-CNN.ipynb` for training and evaluating models on the OASIS dataset.  
2. Use `test_on_diff_dataset.ipynb` to test generalization capabilities on the enhanced Kaggle dataset.  
3. Fine-tune the pre-trained model using `transfer_learning.ipynb`.  
4. The `trained_cnn.pth` file contains the saved model weights for inference or further experimentation.  

## Acknowledgments  
This project utilizes the following datasets:  
- Chugh, “Best Alzheimer MRI dataset (99% accuracy).” Kaggle, 2022, [https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy)  
- Aithal, “OASIS Alzheimer's Detection.” Kaggle, 2023, [https://www.kaggle.com/datasets/ninadaithal/imagesoasis/](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/)  

This project is inspired by state-of-the-art deep learning research for Alzheimer’s detection.
