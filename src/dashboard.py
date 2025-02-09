# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the saved model and encoder
model = joblib.load('RandomForestClassifier.pkl')
# ohe = joblib.load('clean_ohe.pkl')


# Load the dataset
@st.cache_data  # Cache the dataset for faster loading
def load_data():
    return pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


emp_data = load_data()

# Streamlit App Title
st.title("Employee Attrition Prediction Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Page",
                           ["Dataset Overview",
                            "Data Visualization",
                            "Predict Attrition",
                            "Upload File for Prediction"])

# Page 1: Dataset Overview
if options == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("This section provides an overview of the dataset.")

    # Display the first 5 rows of the dataset
    st.subheader("First 5 Rows of the Dataset")
    st.write(emp_data.head())

    # Display basic statistics
    st.subheader("Dataset Statistics")
    st.write(emp_data.describe())

    # Display missing data
    st.subheader("Missing Data")
    st.write(emp_data.isnull().sum())

    # Display duplicate records
    st.subheader("Duplicate Records")
    st.write(f"Number of duplicate records: {emp_data.duplicated().sum()}")

# Page 2: Data Visualization
elif options == "Data Visualization":
    st.header("Data Visualization")
    st.write("This section provides visualizations of the dataset.")

    # Plot Attrition Distribution
    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(x='Attrition', data=emp_data, ax=ax)
    st.pyplot(fig)

    # Plot Age Distribution
    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(emp_data['Age'], kde=True, ax=ax)
    st.pyplot(fig)

    # Convert categorical features to numerical for heatmap
    # Replace 'Gender' column values using manual encoding
    emp_data["Gender"] = emp_data["Gender"].replace({"Female": 0, "Male": 1})

    # Perform LabelEncoder for 'Attrition' column
    le = LabelEncoder()
    emp_data["Attrition"] = le.fit_transform(emp_data['Attrition'])

    # Perform OneHotEncoder for multiple categorical columns
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(emp_data[['Department',
                                              'EducationField',
                                              'JobRole',
                                              'MaritalStatus',
                                              'OverTime',
                                              'BusinessTravel']])

    encoded_df = pd.DataFrame(encoded.toarray(),
                              columns=encoder.get_feature_names_out())
    employee_data = pd.concat([emp_data, encoded_df], axis=1)
    employee_data = employee_data.drop(['Department',
                                        'EducationField',
                                        'JobRole',
                                        'MaritalStatus',
                                        'OverTime',
                                        'BusinessTravel'], axis=1)

    # Plot Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(employee_data.corr(), annot=True, fmt=".1f", ax=ax)
    st.pyplot(fig)

# Page 3: Predict Attrition
elif options == "Predict Attrition":
    st.header("Predict Employee Attrition")
    st.write("This section allows you to predict attrition.")

    # Input fields for new data
    st.subheader("Input Employee Data")
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    gender = st.radio("Gender", ["Female", "Male"])
    department = st.selectbox("Department", emp_data['Department'].unique())
    education_field = st.selectbox(
        "Education Field", emp_data['EducationField'].unique())
    job_role = st.selectbox("Job Role", emp_data['JobRole'].unique())
    marital_status = st.selectbox(
        "Marital Status", emp_data['MaritalStatus'].unique())
    overtime = st.radio("OverTime", ["Yes", "No"])
    monthly_income = st.number_input(
        "Monthly Income", min_value=1000, max_value=20000, value=5000)
    total_working_years = st.number_input(
        "Total Working Years", min_value=0, max_value=40, value=10)

    # Encode categorical features
    gender_encoded = 0 if gender == "Female" else 1
    overtime_encoded = 1 if overtime == "Yes" else 0

    # OneHotEncoding for categorical features
    input_data = {
        'Department': [department],
        'EducationField': [education_field],
        'JobRole': [job_role],
        'MaritalStatus': [marital_status],
        'OverTime': [overtime]
    }
    input_df = pd.DataFrame(input_data)
    encoded_input = ohe.transform(input_df).toarray()

    # Combine all features into a single input array
    numerical_features = [age,
                          gender_encoded,
                          monthly_income,
                          total_working_years]

    final_input = np.concatenate(
        [numerical_features, encoded_input[0]]).reshape(1, -1)

    # Predict button
    if st.button("Predict Attrition"):
        prediction = model.predict(final_input)
        prediction_prob = model.predict_proba(final_input)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("The employee is likely to leave (Attrition: Yes)")
        else:
            st.success("The employee is likely to stay (Attrition: No)")

        st.write(f"Prediction Probability: {prediction_prob[0][1]:.2f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by [Your Name]")
