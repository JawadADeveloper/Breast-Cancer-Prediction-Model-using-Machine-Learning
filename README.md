# Breast Cancer Classification Project

## Overview

This project focuses on classifying breast cancer tumors as malignant or benign using machine learning algorithms. The Breast Cancer Wisconsin (Diagnostic) Dataset is utilized to train and evaluate several models, including Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Gaussian Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, and Stochastic Gradient Descent (SGD). The project includes:

- Data exploration and preprocessing
- Model training and evaluation
- Comparative analysis of different machine learning algorithms
- A web application for model predictions using Flask

## Requirements

- Python 3.7 or higher
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `flask`, `pickle`, `xgboost`, `lightgbm`

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/breast-cancer-classification.git
    cd breast-cancer-classification
    ```

2. **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Required Packages**

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset. It contains various features related to the tumor's characteristics, including:

- `id`: Unique identifier for each patient
- `diagnosis`: Malignant (M) or Benign (B)
- `radius_mean`: Mean radius of the tumor
- `texture_mean`: Mean texture of the tumor
- `perimeter_mean`: Mean perimeter of the tumor
- `area_mean`: Mean area of the tumor
- `smoothness_mean`: Mean smoothness of the tumor
- `compactness_mean`: Mean compactness of the tumor
- `concavity_mean`: Mean concavity of the tumor
- `concave points_mean`: Mean number of concave points of the tumor
- `symmetry_mean`: Mean symmetry of the tumor
- `fractal_dimension_mean`: Mean fractal dimension of the tumor
- `radius_se`: Standard error of mean radius
- `texture_se`: Standard error of mean texture
- `perimeter_se`: Standard error of mean perimeter
- `area_se`: Standard error of mean area
- `smoothness_se`: Standard error of mean smoothness
- `compactness_se`: Standard error of mean compactness
- `concavity_se`: Standard error of mean concavity
- `concave points_se`: Standard error of mean concave points
- `symmetry_se`: Standard error of mean symmetry
- `fractal_dimension_se`: Standard error of mean fractal dimension
- `radius_worst`: Worst radius value
- `texture_worst`: Worst texture value
- `perimeter_worst`: Worst perimeter value
- `area_worst`: Worst area value
- `smoothness_worst`: Worst smoothness value
- `compactness_worst`: Worst compactness value
- `concavity_worst`: Worst concavity value
- `concave points_worst`: Worst number of concave points value
- `symmetry_worst`: Worst symmetry value
- `fractal_dimension_worst`: Worst fractal dimension value
- `Unnamed: 32`: An empty column to be dropped

## Usage

1. **Data Exploration and Preprocessing**

   - Explore and preprocess the dataset to prepare it for training. This includes handling missing values, encoding categorical data, and normalizing features.

2. **Model Training**

   - Train various machine learning models and evaluate their performance using metrics such as accuracy, F1 score, recall, and precision. The models include Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM.

3. **Model Comparison**

   - Compare the performance of different models and visualize the results using bar plots and heatmaps.

4. **Web Application**

   - The `app.py` file contains a Flask application that allows users to input features and get predictions from a trained model. The model is loaded from a pickle file (`model.pkl`).

   To run the Flask app:

    ```bash
    python app.py
    ```

   Open a web browser and navigate to `http://127.0.0.1:5000/` to access the application.
   
### Contribution
Feel free to contribute to this project by submitting issues or pull requests. For detailed contribution guidelines, please refer to the CONTRIBUTING.md file.

## Example

Here's an example of how to train a model and visualize its performance:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# X, y = load_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, 
                              columns=['Predicted: 0', 'Predicted: 1'],
                              index=['Actual: 0', 'Actual: 1'])
plt.figure(figsize=(5, 3))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

