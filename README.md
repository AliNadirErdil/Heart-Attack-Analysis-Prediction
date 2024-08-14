# 🫀 Heart Attack Analysis Prediction

This project focuses on analyzing patient data to predict the likelihood of a heart attack. It begins with a thorough exploratory data analysis (EDA) to understand the dataset's structure and key characteristics. Initial steps include checking for missing values, summarizing statistical properties, and identifying any anomalies or outliers using methods such as the Z-score.

Following the EDA, the project proceeds with detailed data visualization to uncover patterns and relationships. Violin plots are used to examine the distribution of numerical features across different output classes, while pie charts reveal the distribution of categorical features with respect to heart disease. Scatter plots and heatmaps further help in understanding the correlations between features and their impact on the target variable.

Data preprocessing is a crucial step in preparing the dataset for modeling. This involves encoding categorical variables into numerical formats, scaling numerical features using techniques like `RobustScaler` to handle outliers, and splitting the data into training and testing sets to evaluate the model’s performance effectively.

The core of the project involves building and training a logistic regression model. This model is used to predict the probability of a heart attack based on the processed data. The evaluation of the model includes accuracy assessment, detailed classification metrics, and providing a comprehensive classification report to ensure its reliability and effectiveness in predicting heart disease risk.

Overall, the project aims to provide actionable insights into the factors contributing to heart disease and develop a predictive tool that healthcare professionals can use to assess heart attack risk more accurately.



## Table of Contents
- [Dataset](#dataset)
- [Installation & Usage](#installation--usage)
- [Project Overview](#project-overview)
  - [1. Data Loading and Exploration](#1-data-loading-and-exploration)
  - [2. Outlier Detection](#2-outlier-detection)
  - [3. Data Visualization](#3-data-visualization)
  - [4. Correlation Analysis](#4-correlation-analysis)
  - [5. Data Preprocessing](#5-data-preprocessing)
  - [6. Model Training and Evaluation](#6-model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)


## Dataset

The dataset utilized in this project is `heart.csv`, which serves as a comprehensive repository of patient health information. This dataset encompasses a variety of health-related attributes collected from individuals, providing valuable insights into their cardiovascular health. Each row in the dataset represents a single patient and includes several features that capture different aspects of their health status.

Key attributes in the dataset include:

- **Age**: The age of the patient, which is a critical factor in assessing heart disease risk. Older age is generally associated with a higher risk of heart disease.

- **Sex**: The gender of the patient. Gender differences can influence heart disease prevalence, with certain conditions or risk factors being more common in one gender compared to the other.

- **Chest Pain Type (cp)**: The type of chest pain experienced by the patient. This feature is categorized into several types, including:
  - **Typical Angina**: Chest pain due to exertion, often associated with atherosclerosis.
  - **Atypical Angina**: Pain with atypical characteristics, which may not be classic heart-related pain.
  - **Non-Anginal Pain**: Pain that is not related to heart disease.
  - **Asymptomatic**: No chest pain reported.

- **Resting Blood Pressure (trtbps)**: The patient’s blood pressure measured at rest. Elevated blood pressure is a significant indicator of cardiovascular health and a risk factor for heart disease.

- **Cholesterol (chol)**: The serum cholesterol level of the patient. High cholesterol levels are a well-established risk factor for developing heart disease, as they can lead to plaque buildup in the arteries.

- **Fasting Blood Sugar (fbs)**: The level of blood sugar after fasting, which helps assess the risk of diabetes. Elevated fasting blood sugar levels are linked to an increased risk of heart disease due to the effects of diabetes on cardiovascular health.

- **Resting Electrocardiographic Results (restecg)**: Results from an electrocardiogram (ECG) performed at rest. This test helps in identifying various heart conditions and abnormalities:
  - **Normal**: No significant abnormalities detected.
  - **ST-T Wave Abnormality**: Possible indicator of heart disease.
  - **Left Ventricular Hypertrophy**: Thickening of the heart's left ventricle, which can be a sign of underlying cardiovascular issues.

- **Maximum Heart Rate Achieved (thalachh)**: The highest heart rate recorded during exercise. This measure reflects the heart's ability to handle physical stress and can indicate cardiovascular fitness.

- **Depression Induced by Exercise (oldpeak)**: The level of depression (or ST segment depression) induced by exercise relative to the rest phase. This measure helps in identifying potential ischemia or heart-related issues during physical activity.

- **Number of Major Vessels Colored by Fluoroscopy (caa)**: The number of major blood vessels with visible blockages as assessed by fluoroscopy. More vessels with blockages are indicative of a higher risk of heart disease.

- **Thalassemia (thall)**: A blood condition categorized into different types, which affect the overall health profile of the patient:
  - **Normal**: No abnormalities related to thalassemia.
  - **Fixed Defect**: Permanent changes detected, possibly indicating past heart issues.
  - **Reversible Defect**: Changes that are reversible, which might be associated with current ischemic events.

The target variable in the dataset is `output`, which indicates whether the patient has heart disease (`1`) or not (`0`). This binary classification variable is essential for training and evaluating the predictive model, as it allows for the assessment of the model’s ability to distinguish between patients with and without heart disease.

Overall, the `heart.csv` dataset provides a rich source of information for analyzing and predicting heart disease risk, enabling the development of a robust model for heart attack prediction based on various health attributes.

## Installation & Usage

To run this project and execute the provided code, you'll need to set up your environment with the necessary libraries and dependencies. Follow the instructions below to install the required packages.

### Prerequisites

Ensure you have Python installed on your system. It is recommended to use Python 3.6 or higher. You can download Python from the [official Python website](https://www.python.org/downloads/).

### Step-by-Step Installation

1. **Clone the Repository**

   First, clone this repository to your local machine using Git. Open your terminal or command prompt and run the following command:

```bash
git clone https://github.com/yourusername/heart-attack-analysis.git
```
2. **Navigate to the Project Directory**

Change your directory to the project folder:
   ```bash
cd heart-attack-analysis
   ```
3. **Open the Jupyter Notebook**
```bash
jupyter notebook
```

# Project Overview

## 1. Data Loading and Exploration
- **Data Loading**: The dataset is loaded using the `pandas` library. Initial exploration includes checking the data shape, data types, and missing values.
- **Basic Statistics**: Descriptive statistics are generated to understand the distribution and central tendencies of the dataset.
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('heart.csv')

# Basic data exploration
print(df.shape)
print(df.info())
print(df.describe().T)
```
![image](https://github.com/user-attachments/assets/518abd0b-f1bd-43c5-8b43-8b3757b3e25e)

## 2. Outlier Detection
- **Z-Score Method**: Outliers are detected using the `Z-score method`. Rows with Z-scores greater than 3 in any numerical feature are identified as potential outliers.

- **Outlier Rows**: After detecting outliers, the specific rows containing these outliers are examined to understand their impact on the dataset.

```python
import numpy as np
from scipy import stats

# Select numerical columns
num_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

# Calculate Z-scores
z_scores = np.abs(stats.zscore(df[num_cols]))

# Identify outliers
outliers = (z_scores > 3).any(axis=1)
outlier_rows = df[outliers]

print("Rows with potential outliers:")
print(outlier_rows)
```

## 3. Data Visualization
- **Violin Plots**: Violin plots are utilized to visualize the distribution of numerical features across the two classes defined by the output variable (0 and 1). Each plot provides a combined view of the data distribution's probability density and cumulative distribution. By comparing these plots for different features, one can observe how the distribution varies between patients who have heart disease and those who do not. This visualization helps in understanding whether certain numerical features exhibit distinct patterns or variations associated with the presence or absence of heart disease.
![violin_plots](https://github.com/user-attachments/assets/cb60023f-ff3f-4118-96c7-57a9c010cc7c)

- **Pie Charts**: Pie charts are created for categorical features to display the distribution of each category in relation to the output variable. These charts visually represent the proportion of each category within the context of the `heart disease classification`, making it easier to understand how different categorical attributes are associated with heart disease. For example, pie charts might show the proportion of patients with different chest pain types or fasting blood sugar levels, illustrating how these categories contribute to the likelihood of heart disease.

![image](https://github.com/user-attachments/assets/f7aeccc7-b7bf-45cf-a24b-9bd64e4245ac)

- **Scatter Plots**: Scatter plots are employed to explore the relationship between numerical features and the output variable. Each plot displays data points representing individual patients, with color-coding used to differentiate between the two classes of the output variable (0 and 1). Scatter plots provide a visual way to assess `correlations`, `clusters`, and potential patterns within the data, helping to identify how specific numerical features relate to the presence or absence of heart disease.

![numeric_feature_scatter_plots](https://github.com/user-attachments/assets/2f26d73e-c4a1-4400-8723-a3bf841f583e)

- **Heatmaps**: Correlation heatmaps are used to visualize the relationships between different numerical features in the dataset. By displaying a matrix of correlation coefficients, heatmaps highlight the degree to which features are linearly related to each other. This visualization aids in identifying features with strong correlations, which can be crucial for feature selection and understanding multicollinearity. High correlation between features may indicate redundancy, while low correlation can suggest unique contributions to the predictive model.

![image](https://github.com/user-attachments/assets/d1ad4c44-bfa6-497c-af09-eedb0a88ef30)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Violin plots for numerical features
for feature in num_cols:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='output', y=feature, data=df)
    plt.title(f'{feature} Distribution by Output')
    plt.xlabel('Output')
    plt.ylabel(feature)
    plt.grid(True)
    plt.show()

# Pie Charts for categorical features
categorical_features = {
    'sex': {0: 'Male', 1: 'Female'},
    'cp': {1: 'Typical angina', 2: 'Atypical angina', 3: 'Non-anginal pain', 4: 'Asymptomatic'},
    'fbs': {0: 'Less than or equal to 120 mg/dl', 1: 'Greater than 120 mg/dl'},
    'restecg': {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'},
    'exng': {0: 'No Exercise Induced Angina', 1: 'Exercise Induced Angina'},
    'slp': {0: 'Low slope', 1: 'Normal slope', 2: 'High slope'},
    'caa': {0: '0 vessels', 1: '1 vessel', 2: '2 vessels', 3: '3 vessels'},
    'thall': {1: 'Normal', 2: 'Fixed defect', 3: 'Reversible defect'}
}

for feature, labels in categorical_features.items():
    feature_counts = df.groupby([feature, 'output']).size().unstack().fillna(0)
    existing_categories = feature_counts.index.tolist()
    
    for category, label in labels.items():
        if category in existing_categories:
            plt.figure(figsize=(8, 8))
            category_counts = feature_counts.loc[category]
            plt.pie(category_counts, labels=['Not Heart Attack', 'Heart Attack'], autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'])
            plt.title(f'{feature} - {label} Distribution by Output')
            plt.show()
```

## 4. Correlation Analysis

- **Correlation Matrix**: A correlation matrix is computed to understand the relationships between the different numerical features. This step is crucial for identifying multicollinearity and understanding which features are most strongly related to the target variable output.

- **Heatmap Visualization**: The correlation matrix is visualized using a heatmap, providing a clear and intuitive understanding of the feature relationships.

```python
# Correlation matrix
correlation_matrix = df.corr()

# Heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16)
plt.show()
```
![image](https://github.com/user-attachments/assets/4df6240b-2d5e-4a14-9ab6-d7dc4eee5703)

## 5. Data Preprocessing


- **Feature Encoding**: Categorical features are one-hot encoded to convert them into a format suitable for machine learning models.

- **Feature Scaling**: Numerical features are scaled using `RobustScaler` to handle any potential outliers and ensure that the data is normalized before feeding it into the model.

- **Train-Test Split**: The dataset is split into training and testing sets to evaluate the model's performance on unseen data.


```python
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Encoding categorical features and scaling numerical features
df_encoded = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall'], drop_first=True)
scaler = RobustScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# Split data into training and testing sets
X = df_encoded.drop('output', axis=1)
y = df_encoded['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## 6. Model Training and Evaluation

- **Logistic Regression**: A logistic `regression model` is trained on the preprocessed data. This model is chosen for its simplicity and effectiveness in binary classification tasks.

- **Model Evaluation**: The performance of the `logistic regression model` is assessed using several metrics:

  - **Accuracy**: This metric measures the proportion of correct predictions made by the model. It indicates the overall success rate of the model in classifying instances correctly.

  - **Classification Report**: This report provides detailed metrics for evaluating the model's performance, including:
    - **Precision**: The proportion of true positive predictions among all positive predictions made by the model. It reflects how many of the predicted positive cases are actually positive.
    - **Recall**: The proportion of true positive predictions among all actual positive cases. It measures how well the model identifies all relevant cases.
    - **F1-Score**: The harmonic mean of precision and recall, providing a single metric that balances both precision and recall, especially useful when dealing with imbalanced classes.
    - **Support**: The number of actual occurrences of each class in the dataset, which provides context for the precision, recall, and F1-score values.

These evaluation metrics are crucial for understanding the model's performance in predicting heart disease and ensuring its reliability in making accurate predictions.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Train logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Make predictions
y_pred = logreg.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```
# Results

- **Accuracy**: The logistic regression model achieved an accuracy score of 0.9016 on the test set.
- **Classification Report**: The classification report shows precision, recall, and F1-score for both classes (Less chance of heart attack and More chance of heart attack).
![image](https://github.com/user-attachments/assets/52b3d2bc-9ee2-4d7a-9a12-cd8bf8733575)


# Conclusion
This project illustrates the methodology for predicting heart disease using `logistic regression`, a fundamental technique in binary classification problems. Here’s a detailed summary of the findings and implications:

1. **Model Performance:**
   - The logistic regression model has shown promising performance on the current dataset. Evaluation metrics, such as accuracy, precision, recall, and F1 score, indicate that the model effectively differentiates between patients with and without heart disease.
   - The use of a confusion matrix  confirms the model’s ability to handle the classification task with a satisfactory level of performance.

2. **Feature Analysis:**
   - Key features such as `age`, `blood pressure`, `cholesterol levels`, `maximum heart rate`, and `depression induced by exercise` have been found to be significant predictors of heart disease.
   - Visualizations, including violin plots, reveal how these features vary across different output classes, providing insights into the model's interpretability and feature importance.

3. **Model Limitations:**
   - Despite its effectiveness, the model's performance could be constrained by the dataset’s size and diversity. The dataset used may not fully represent the broader population, which could impact the model’s generalizability.
   - There is a possibility of overfitting or underfitting due to the choice of hyperparameters and the logistic regression algorithm's simplicity.

4. **Future Work:**
   - To enhance the model’s robustness and accuracy, further hyperparameter tuning and experimentation with more advanced algorithms, such as ensemble methods or neural networks, are recommended.
   - Validation on additional and more diverse datasets will be crucial for assessing the model’s generalizability and ensuring its applicability in real-world scenarios.
   - Incorporating more features and employing feature engineering techniques could also improve the model's predictive power.

In summary, while the logistic regression model has demonstrated strong performance on the given dataset, ongoing improvements through validation, tuning, and the use of diverse datasets are essential for refining the model and achieving better predictive accuracy for heart disease.


