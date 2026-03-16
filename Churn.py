# -------------------------------
# Customer Churn Prediction Dashboard
# -------------------------------

# 1. Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load model, feature columns, and dataset
model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
df = pd.read_csv("Churn_Modelling.csv")  # original dataset

# Preprocess dataset (same as training)
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
df = pd.get_dummies(df, columns=['Geography','Gender'], drop_first=True)

# 3. Sidebar for Inputs
st.sidebar.header("Enter Customer Details:")

credit_score = st.sidebar.number_input("Credit Score", 300, 850, 650)
age = st.sidebar.number_input("Age", 18, 100, 30)
tenure = st.sidebar.number_input("Tenure (years with bank)", 0, 10, 3)
balance = st.sidebar.number_input("Balance ($)", 0.0, 250000.0, 50000.0)
num_products = st.sidebar.number_input("Number of Products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 1000.0, 200000.0, 50000.0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

# 4. Prepare input dataframe
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [1 if has_cr_card=="Yes" else 0],
    'IsActiveMember': [1 if is_active_member=="Yes" else 0],
    'EstimatedSalary': [estimated_salary],
    'Gender_Male': [1 if gender=="Male" else 0],
    'Geography_Germany': [1 if geography=="Germany" else 0],
    'Geography_Spain': [1 if geography=="Spain" else 0]
})

# Ensure columns match training
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

# 5. Prediction
st.header("🔮 Customer Churn Prediction")
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result:")
    st.write("🔴 Churn" if prediction==1 else "🟢 No Churn")
    st.write(f"Probability of churn: {probability*100:.2f}%")

# 6. Dataset Preview
st.header("📊 Dataset Preview")
if st.checkbox("Show Dataset"):
    st.dataframe(df.head(20))  # show first 20 rows

# 7. Feature Importance
st.header("🌟 Feature Importance ")
importances = model.feature_importances_
feat_names = feature_columns
importance_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
plt.title("Top Features Affecting Churn")
st.pyplot(fig)

# 8. Model Accuracy Chart with Annotations
st.header("📈 Model Accuracy Comparison")

# Accuracy values
accuracy_dict = {
    "Logistic Regression": 0.811,
    "Random Forest": 0.8665,
    "XGBoost": 0.8695
}

accuracy_df = pd.DataFrame(list(accuracy_dict.items()), columns=["Model", "Accuracy"])

# Plot
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="Model", y="Accuracy", data=accuracy_df, palette="magma", ax=ax)
plt.ylim(0,1)
plt.title("Model Accuracy Comparison")

# Annotate bars with accuracy values
for i, row in accuracy_df.iterrows():
    ax.text(i, row.Accuracy + 0.01, f"{row.Accuracy:.4f}", ha='center', va='bottom', fontweight='bold')

st.pyplot(fig)
