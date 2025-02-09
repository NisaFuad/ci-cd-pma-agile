# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the best model and encoder
model = joblib.load('GradientBoostingClassifier.pkl')


# Load the dataset
@st.cache_data  # Cache the dataset for faster loading
def load_data():
    return pd.read_csv("data/HR_Dataset.csv")


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

    # Remove duplicate rows
    st.write(f"Dataset shape after before duplicates: {emp_data.shape}")
    emp_data = emp_data.drop_duplicates()
    st.write(f"Dataset shape after removing duplicates: {emp_data.shape}")

# Page 2: Data Visualization
elif options == "Data Visualization":
    st.header("Data Visualization")
    st.write("This section provides visualizations of the dataset.")

    # Plot Attrition Distribution
    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(x='left', data=emp_data, ax=ax)
    st.pyplot(fig)

    # Plot Age Distribution
    st.subheader("Time Spent in Company Distribution")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(emp_data['time_spend_company'], kde=True, ax=ax)
    st.pyplot(fig)

    # Plot Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))

    # creating labelEncoder
    le = LabelEncoder()

    # Converting string labels into numbers.
    salary_encoded = le.fit_transform(emp_data['salary'])
    dept_encoded = le.fit_transform(emp_data['Departments'])

    emp_data_encoded = emp_data.drop(['Departments', 'salary'], axis=1)

    # Add the encoded columns back to the dataset
    emp_data_encoded['salary_encoded'] = salary_encoded
    emp_data_encoded['dept_encoded'] = dept_encoded

    sns.heatmap(emp_data_encoded.corr(), annot=True, fmt=".1f", ax=ax)
    st.pyplot(fig)

# Page 3: Predict Attrition
elif options == "Predict Attrition":
    st.header("Predict Employee Attrition")
    st.write("This section allows you to predict attrition.")

    # Input fields for new data
    st.subheader("Input Employee Data")
    # age = st.number_input("Age", min_value=18, max_value=65, value=30)
    department = st.selectbox("Department", emp_data['Departments'].unique())
    salary = st.selectbox("Salary Level", emp_data['salary'].unique())
    promotion = st.radio("Promotion in the last 5 Years", ["Yes", "No"])
    time_spent = st.number_input("Total Working Years",
                                 min_value=2, max_value=10, value=5)
    monthly_hr = st.number_input("Monthly Hour",
                                 min_value=96, max_value=310, value=100)
    number_project = st.number_input("Total Projects Involved",
                                     min_value=2, max_value=7, value=5)
    work_accident = st.radio("Work Accident", ["Yes", "No"])
    last_performance = st.number_input("Last performance Evaluation (0 to 1)",
                                       min_value=0.36,
                                       max_value=1.0,
                                       value=0.5)
    satistaction = st.number_input("Satisfaction level (0 to 1)",
                                   min_value=0.09, max_value=1.0, value=0.5)

    # Encode categorical features
    work_accident_encoded = 0 if work_accident == "Yes" else 1
    promotion_encoded = 1 if promotion == "Yes" else 0

    # Combine all features into a single input array
    numerical_features = [time_spent,
                          monthly_hr,
                          number_project,
                          last_performance,
                          satistaction,
                          work_accident_encoded,
                          promotion_encoded]

    # creating labelEncoder
    le = LabelEncoder()

    # Converting string labels into numbers.
    salary_encoded = le.fit_transform([department])[0]
    dept_encoded = le.fit_transform([salary])[0]

    final_input = np.concatenate((np.array([dept_encoded]),
                                  np.array([salary_encoded]),
                                  np.array(numerical_features)))

    processed_data = pd.read_csv("data/processed_df.csv")
    feature_names = processed_data.drop(columns=['left']).columns

    # Convert final_input with the same column names as the training data
    final_input_df = pd.DataFrame(final_input.reshape(1, -1),
                                  columns=feature_names)

    # Predict button
    if st.button("Predict Attrition"):
        prediction = model.predict(final_input_df)
        prediction_prob = model.predict_proba(final_input_df)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("The employee is likely to leave (Attrition: Yes)")
        else:
            st.success("The employee is likely to stay (Attrition: No)")

        st.write(f"Prediction Probability: {prediction_prob[0][1]:.2f}")

# Page 4: Upload File for Prediction
elif options == "Upload File for Prediction":
    st.header("Upload File for Prediction")
    st.write("This section allows you to predict employee attrition based on file uploaded.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded file
        uploaded_data = pd.read_csv(uploaded_file)

        # Display the first few rows of the uploaded file
        st.subheader("Uploaded File Preview")
        st.write(uploaded_data.head())

        # Preprocess the uploaded data
        uploaded_data_encoded = uploaded_data.copy()

        # Encoding categorical columns in the uploaded data
        for col in ['Departments', 'salary']:
            le = LabelEncoder()
            uploaded_data_encoded[col] = le.fit_transform(
                uploaded_data_encoded[col])

        # Drop any columns that are not part of the prediction
        uploaded_data_encoded = uploaded_data_encoded.drop(['left'],
                                                           axis=1,
                                                           errors='ignore')

        # Make predictions on the uploaded data
        predictions = model.predict(uploaded_data_encoded)
        prediction_probs = model.predict_proba(uploaded_data_encoded)

        # Display the prediction results
        uploaded_data['Prediction'] = predictions
        uploaded_data['Prediction Probability'] = prediction_probs[:, 1]

        st.subheader("Prediction Results")
        st.write(uploaded_data)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by Nor Hanisah Ahmed Fuad(MRT231005)")
