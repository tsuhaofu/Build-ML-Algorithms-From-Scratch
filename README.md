# Build ML Algorithms From Scratch

## Overview
This repository contains five distinct data science projects, each focusing on a different machine learning technique applied to various datasets. These projects encompass techniques from image processing to classification and clustering, demonstrating the application of fundamental algorithms in solving real-world problems.

### Projects List:
1. **SVD Image Compression and Reconstruction**
2. **Gaussian Naive Bayes Classifier on the MNIST Dataset**
3. **One-vs-All Logistic Regression Ensemble on the MNIST Dataset**
4. **K-means Clustering on the Iris Dataset**
5. **Color Quantization using K-means**

---

## 1. SVD Image Compression and Reconstruction

### Objective
To implement Singular Value Decomposition (SVD) from scratch and utilize it for compressing and reconstructing an image, aiming to understand the trade-offs between image quality and data reduction.

### Methodology
- Decomposed images into RGB channels and performed SVD on each channel separately.
- Explored different numbers of singular values to examine their impact on the image quality.
- Calculated Mean Squared Error (MSE) to quantify the loss from the original image.

### Results
- Demonstrated effective image compression and reconstruction with variable numbers of singular values.
- Visual comparisons and MSE calculations provided insights into the optimal balance between compression and quality.

---

## 2. Gaussian Naive Bayes Classifier on the MNIST Dataset

### Objective
To develop a Gaussian Naive Bayes classifier from scratch and apply it to classify handwritten digits from the MNIST dataset.

### Methodology
- Implemented Gaussian distribution calculations for each class.
- Applied smoothing techniques to avoid numerical instability.
- Evaluated model performance across different smoothing parameters using accuracy and a confusion matrix.

### Results
- Achieved competitive accuracies on the MNIST test set.
- Identified optimal smoothing parameter for the best balance between precision and computational efficiency.

---

## 3. One-vs-All Logistic Regression Ensemble on the MNIST Dataset

### Objective
Build a series of logistic regression models in a one-vs-all setup for the MNIST dataset and combine them into an ensemble to improve prediction accuracy.

### Methodology
- Developed ten logistic regression models, each predicting the likelihood of a digit.
- Combined predictions using ensemble techniques to make final class decisions.

### Results
- Each model's performance and the ensemble's overall accuracy were reported.
- The ensemble model demonstrated superior performance compared to individual classifiers.

---

## 4. K-means Clustering on the Iris Dataset

### Objective
To implement K-means clustering from scratch and apply it to the Iris dataset to identify distinct groups based on flower characteristics.

### Methodology
- Conducted clustering with varying numbers of clusters.
- Used PCA for dimensionality reduction to visualize clusters.
- Compared clustering results against the known labels from the Iris dataset.

### Results
- Evaluated the effectiveness of the clustering by comparing predicted clusters with true labels.
- Discussed any discrepancies and potential reasons for poor clustering performance in specific cases.

---

## 5. Color Quantization using K-means

### Objective
To reduce the number of distinct colors in an image using K-means clustering, aiming to maintain as much of the image's visual quality as possible.

### Methodology
- Clustering pixel values to reduce color variance while retaining visual similarity.
- Applied different numbers of clusters to observe the impact on image quality and file size.

### Results
- Successfully demonstrated color quantization with various palette sizes.
- Visual and quantitative analysis provided to show the trade-offs involved in color reduction.

---

## Installation and Usage
Instructions for setting up the environment and running the projects can be found in the respective project folders. Each folder contains detailed steps and scripts to execute the projects.

---

## Conclusion
This portfolio highlights my hands-on approach to developing machine learning algorithms from scratch, demonstrating a deep understanding of their underlying principles. These foundational projects serve as a robust basis for advanced problem-solving in the field of data science.

---
