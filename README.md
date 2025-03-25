
# KNN & PCA Implementation

This repository contains implementations of K-Nearest Neighbors (KNN) and Principal Component Analysis (PCA) using Python and Scikit-Learn. The notebook covers various experiments and comparisons to analyze the performance of these algorithms in classification and regression tasks.

---

## 📌 **Table of Contents**

1. **KNN Classifier**
   - Train KNN classifier on the Iris dataset and evaluate accuracy.
   - Train KNN classifier using different distance metrics (Euclidean & Manhattan) and compare accuracy.
   - Train KNN classifier with different values of K and visualize decision boundaries.
   - Train KNN classifier with `uniform` vs `distance` weights and compare accuracy.
   - Train KNN classifier using KD Tree and Ball Tree algorithms and compare performance.
   - Train KNN classifier and evaluate using Precision, Recall, and F1-Score.
   - Train KNN classifier with different `leaf_size` values and compare accuracy.
   - Train KNN classifier on the Wine dataset and print the classification report.
   - Train KNN classifier and evaluate using ROC-AUC score.
   - Train KNN classifier and perform feature selection before training.
   - Train KNN classifier and visualize the decision boundary.

2. **KNN Regressor**
   - Train KNN regressor on a synthetic dataset and evaluate using Mean Squared Error (MSE).
   - Train KNN regressor and analyze the effect of different K values on performance.
   - Train KNN regressor and analyze the effect of different distance metrics on prediction error.

3. **Feature Scaling & KNN**
   - Apply feature scaling before training a KNN model and compare results with unscaled data.

4. **PCA (Principal Component Analysis)**
   - Train a PCA model on synthetic data and print the explained variance ratio.
   - Apply PCA before training a KNN classifier and compare accuracy with and without PCA.
   - Train a PCA model and visualize the cumulative explained variance.
   - Train a PCA model and visualize data projection onto the first two principal components.
   - Train a PCA model on a high-dimensional dataset and visualize the Scree plot.
   - Train a PCA model and analyze the effect of different numbers of components on accuracy.
   - Train a PCA model and visualize how data points transform before and after PCA.
   - Train a PCA model and visualize the variance captured by each principal component.
   - Train a PCA model and visualize the data reconstruction error after reducing dimensions.
   - Train a PCA model and analyze the effect of different numbers of components on data variance.

5. **KNN & PCA Together**
   - Perform hyperparameter tuning on a KNN classifier using GridSearchCV.
   - Train a PCA model and use KNN for classification, comparing results with and without PCA.
   - Train a KNN classifier and check the number of misclassified samples.

6. **Missing Values & KNN**
   - Implement KNN imputation for handling missing values in a dataset.
   - Train a KNN classifier and analyze how it handles missing values.

---

## 📂 **Installation & Setup**

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/KNN-PCA-Analysis.git
   cd KNN-PCA-Analysis
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```sh
   jupyter notebook
   ```

---

## 🛠 **Technologies Used**
- Python 🐍
- Scikit-Learn 🤖
- Matplotlib 📊
- Seaborn 🎨
- NumPy 🔢
- Pandas 🗂

---

## 📌 **Results & Visualizations**
- Various decision boundaries and performance metrics of KNN classifiers.
- Scree plots and cumulative variance graphs for PCA.
- Visualization of PCA-transformed data before and after dimensionality reduction.
- KNN model performance comparison across different hyperparameters and distance metrics.

---

## 📢 **Contributing**
Feel free to fork the repository, open issues, or submit pull requests. Contributions are welcome! 🚀

---

## 📜 **License**
This project is licensed under the MIT License.

---

### 🔥 **Happy Coding! 🚀**

