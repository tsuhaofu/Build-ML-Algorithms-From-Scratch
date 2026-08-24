# Build ML Algorithms From Scratch

## Overview
This repository contains five distinct data science projects, each focusing on a different machine learning technique applied to various datasets. These projects encompass techniques from image processing to classification and clustering, demonstrating the application of fundamental algorithms in solving real-world problems.

### Projects List:
1. **SVD Image Compression and Reconstruction**
2. **Gaussian Naive Bayes Classifier on the MNIST Dataset**
3. **One-vs-All Logistic Regression Ensemble on the MNIST Dataset**
4. **K-means Clustering on the Iris Dataset**
5. **Color Quantization using K-means** — lives inside `K-means/K-means.ipynb`, not a separate folder

### Notebooks
| # | Notebook |
|---|---|
| 1 | `SVD Image Compression and Reconstruction/Singular Value Decomposition.ipynb` |
| 2 | `Gaussian Naive Bayes Classifier on the MNIST Dataset/Gaussian Naive Bayes Classifier.ipynb` |
| 3 | `One-vs-All Logistic Regression Ensemble on the MNIST Dataset/One-vs-All Logistic Regression Ensemble.ipynb` |
| 4, 5 | `K-means/K-means.ipynb` |

Each folder also holds a PDF export of its notebook. The two MNIST projects read
from the `data.zip` in their own folder; the SVD and K-means projects use the
`.jpg` alongside them. No external download is required.

### Requirements
`numpy`, `scipy`, `matplotlib`, `scikit-learn`, `Pillow`, `jupyter`.
The algorithms themselves are implemented from scratch — scikit-learn is used
only for PCA in the K-means notebook and for the train/test split.

---

## 1. SVD Image Compression and Reconstruction

### Objective
To implement Singular Value Decomposition (SVD) from scratch and utilize it for compressing and reconstructing an image, aiming to understand the trade-offs between image quality and data reduction.

### Methodology
- Decomposed images into RGB channels and performed SVD on each channel separately.
- Explored different numbers of singular values to examine their impact on the image quality.
- Calculated Mean Squared Error (MSE) to quantify the loss from the original image.

### Results
- Reconstructed at k = 1, 2, 4, 8, 16, 64, 256 and 1080 singular values. MSE against
  the original falls from 100.99 at k=1 to 0.63 at k=1080.
- Most of the signal sits in the leading components: cumulative explained variance is
  already above 0.90 at the first component on all three channels and has flattened
  well before 200.

![Scree plots per RGB channel](figures/svd-scree-plot.png)
*Cumulative explained variance for the red, green and blue channels.*

<details>
<summary>Rank-k reconstructions, each panel labelled with its MSE (2.3 MB image)</summary>

![Rank-k reconstructions with MSE](figures/svd-reconstruction.png)

</details>

---

## 2. Gaussian Naive Bayes Classifier on the MNIST Dataset

### Objective
To develop a Gaussian Naive Bayes classifier from scratch and apply it to classify handwritten digits from the MNIST dataset.

### Methodology
- Implemented Gaussian distribution calculations for each class.
- Applied smoothing techniques to avoid numerical instability.
- Evaluated model performance across different smoothing parameters using accuracy and a confusion matrix.

### Results
- Smoothing swept over {0.001, 0.01, 0.1, 1, 10, 100}. Test accuracy across the sweep
  ranges from 78.38% to 80.80%, i.e. one-zero error 21.62% down to 19.20%.
- The confusion matrix shows the errors are not spread evenly: 5→3 (135), 9→7 (122),
  4→9 (116) and 8→1 (109) account for much of the loss.

![Confusion matrix on the MNIST test set](figures/naive-bayes-confusion-matrix.png)
*Ten-class confusion matrix for the best smoothing setting.*

---

## 3. One-vs-All Logistic Regression Ensemble on the MNIST Dataset

### Objective
Build a series of logistic regression models in a one-vs-all setup for the MNIST dataset and combine them into an ensemble to improve prediction accuracy.

### Methodology
- Developed ten logistic regression models, each predicting the likelihood of a digit.
- Combined predictions using ensemble techniques to make final class decisions.

### Results
- Per-class one-vs-rest accuracy ranges from 96.11% (class 8) to 99.36% (class 1).
- The combined ten-class ensemble reaches 92.12% accuracy, one-zero error 7.88%.
- Note these two figures are not directly comparable: the per-class numbers are binary
  one-vs-rest accuracies, the ensemble number is over all ten classes.

---

## 4. K-means Clustering on the Iris Dataset

### Objective
To implement K-means clustering from scratch and apply it to the Iris dataset to identify distinct groups based on flower characteristics.

### Methodology
- Conducted clustering with varying numbers of clusters.
- Used PCA for dimensionality reduction to visualize clusters.
- Compared clustering results against the known labels from the Iris dataset.

### Results
- Five random restarts at |C|=3, plotted in PCA space against the known Iris labels.
- Four restarts converge to the same partition; one does not — a plain illustration of
  K-means' sensitivity to initialisation rather than a defect in the implementation.

![Five K-means restarts on Iris](figures/kmeans-clusters.png)
*Five random restarts at |C|=3 in PCA space, centroids marked in red.*

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
```bash
pip install numpy scipy matplotlib scikit-learn Pillow jupyter
jupyter notebook
```
Open any of the notebooks listed above and run it top to bottom. The MNIST projects
expect `data.zip` unzipped in place alongside the notebook.

---

## Conclusion
This portfolio highlights my hands-on approach to developing machine learning algorithms from scratch, demonstrating a deep understanding of their underlying principles. These foundational projects serve as a robust basis for advanced problem-solving in the field of data science.

---
