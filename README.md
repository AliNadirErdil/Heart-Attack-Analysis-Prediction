# 🫀 Heart Attack Analysis Prediction

This project focuses on analyzing patient data to predict the likelihood of a heart attack. It begins with a thorough exploratory data analysis (EDA) to understand the dataset's structure and key characteristics. Initial steps include checking for missing values, summarizing statistical properties, and identifying any anomalies or outliers using methods such as the Z-score.

Following the EDA, the project proceeds with detailed data visualization to uncover patterns and relationships. Violin plots are used to examine the distribution of numerical features across different output classes, while pie charts reveal the distribution of categorical features with respect to heart disease. Scatter plots and heatmaps further help in understanding the correlations between features and their impact on the target variable.

Data preprocessing is a crucial step in preparing the dataset for modeling. This involves encoding categorical variables into numerical formats, scaling numerical features using techniques like RobustScaler to handle outliers, and splitting the data into training and testing sets to evaluate the model’s performance effectively.

The core of the project involves building and training a logistic regression model. This model is used to predict the probability of a heart attack based on the processed data. The evaluation of the model includes accuracy assessment, detailed classification metrics, and the ROC-AUC score to ensure its reliability and effectiveness in predicting heart disease risk.

Overall, the project aims to provide actionable insights into the factors contributing to heart disease and develop a predictive tool that healthcare professionals can use to assess heart attack risk more accurately.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
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




