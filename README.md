# Task 1: End-to-End ML Project with Streamlit GUI (Customer Churn Prediction)

## Overview
This repository contains the solution for Task 1: End-to-End ML Project with Streamlit GUI, completed as part of my AI & Data Science Internship tasks.  

The project focuses on building a complete **machine learning pipeline** for predicting **customer churn** and deploying the trained model through an **interactive Streamlit dashboard**.

---

## Objective
The objective of this task is to:

- Build a complete **end-to-end ML pipeline** for customer churn prediction.
- Preprocess and clean real-world banking data.
- Train and evaluate multiple models (Logistic Regression, Random Forest, XGBoost) to identify the best-performing model.
- Deploy the model using a **Streamlit interactive dashboard** with prediction, probability, dataset preview, feature importance, and model performance visualization.

This task emphasizes both **data science workflow** and **real-world application deployment**.

---

## Dataset
**Dataset Name:** Customer Churn Dataset  
**Type:** Structured CSV dataset  

### Dataset Description
- Total Records: **10,000+**  
- Target Variable: **Exited**
  - 1 → Customer left the bank (Churn)  
  - 0 → Customer stayed with the bank  
- Features:
  - CreditScore – Credit score of the customer
  - Geography – Country where the customer resides
  - Gender – Gender of the customer
  - Age – Age of the customer
  - Tenure – Number of years with the bank
  - Balance – Account balance
  - NumOfProducts – Number of bank products used
  - HasCrCard – Whether the customer owns a credit card
  - IsActiveMember – Whether the customer is active
  - EstimatedSalary – Estimated annual salary

The dataset is widely used to **analyze factors influencing customer churn** and build predictive machine learning models.

---

## Tools & Technologies Used
- Python  
- pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- scikit-learn  
- XGBoost  
- Joblib  
- Streamlit  
- Jupyter Notebook / Google Colab  

---

## Approach

### 1️⃣ Data Loading
- Loaded the dataset using `pandas.read_csv()`  
- Converted CSV into a pandas DataFrame for easy manipulation  

### 2️⃣ Data Inspection
- `.shape` → check dataset dimensions  
- `.info()` → inspect data types and missing values  
- `.columns` → view column names  
- `.head()` → preview first few rows  

### 3️⃣ Data Cleaning & Preprocessing
- Checked for missing values with `.isnull().sum()` → none found  
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`  
- Encoded categorical variables using **One-Hot Encoding** (`Gender`, `Geography`)  
- Standardized numeric features for models like Logistic Regression  

### 4️⃣ Exploratory Data Analysis (EDA)
- Target distribution plot → observed class imbalance  
- Age vs Churn histogram → older customers more likely to churn  
- Correlation heatmap → identified strong predictors (Age, Balance, IsActiveMember)  
- Feature importance visualization using Random Forest → confirmed top features  

### 5️⃣ Train-Test Split
- Split dataset into **80% training** and **20% testing**  
- Ensured model evaluation is performed on unseen data  

### 6️⃣ Model Training & Evaluation
- Trained three models:
  - Logistic Regression → 0.811 accuracy  
  - Random Forest → 0.8665 accuracy  
  - XGBoost → 0.8695 accuracy (**best model**)  
- Evaluated models using:
  - Accuracy Score
  - Classification Report  

### 7️⃣ Model Saving
- Saved **best-performing XGBoost model** using Joblib  
- Saved **scaler object** for consistent input preprocessing  
- Saved **feature column names** to maintain correct input structure in the dashboard  

### 8️⃣ Streamlit Dashboard
- Interactive sidebar for user inputs  
- Predicts churn probability and class (Churn / No Churn)  
- Displays dataset preview (first 20 rows)  
- Shows **feature importance** and **model accuracy comparison**  

---

## Results & Insights
- XGBoost achieved the **highest accuracy (0.8695)**, making it the most suitable model for deployment  
- Feature importance shows that **Age, Balance, IsActiveMember, and NumOfProducts** are strong churn predictors  
- Dataset imbalance exists; handling class weights or resampling can improve performance  
- Interactive dashboard allows real-time predictions and visualizations for better decision-making  

---

## Conclusion
This project demonstrates a **complete end-to-end ML workflow**:

- Data loading, inspection, cleaning, and preprocessing  
- Exploratory Data Analysis (EDA) with insights  
- Model training, evaluation, and selection of the best model  
- Deployment of the ML model using an **interactive Streamlit dashboard**  

The XGBoost model provides **accurate and reliable churn predictions**, and the dashboard offers a practical tool for real-world business use cases.

---

## Skills Gained
- End-to-end ML pipeline development  
- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Model training and evaluation with multiple algorithms  
- Feature importance analysis  
- Model deployment using Streamlit  
- Joblib for model and scaler persistence  
- Visualization and dashboard creation for interactive user experience
