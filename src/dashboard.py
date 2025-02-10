# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the best model and encoder
model = joblib.load('GradientBoostingClassifier.pkl')


# Load the dataset
@st.cache_data  # Cache the dataset for faster loading
def load_data():
    return pd.read_csv("data/HR_Dataset.csv")


emp_data = load_data()


# Load column information
@st.cache_data
def load_column_info():
    return pd.read_csv("data/column_info.csv")


column_info = load_column_info()

# Streamlit App Title
st.title("Employee Attrition Prediction Dashboard")

# Custom CSS for background and sidebar styling
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background-color: #f0f0f0; /* Light gray */
        color: #333333; /* Dark gray text */
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #3a3a3a !important; /* Dark gray */
        color: #ffffff !important; /* White text */
    }

    /* Sidebar Text */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Title Styling */
    .stTitle {
        color: #444444; /* Slightly darker gray for contrast */
        font-weight: bold;
    }

    /* Buttons */
    .stButton>button {
        background-color: #444444; /* Dark gray */
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #333333; /* Slightly darker on hover */
    }

    /* DataFrame Styling */
    .stDataFrame {
        border: 2px solid #888888;
        border-radius: 5px;
    }

        /* Apply to all pages */
    .stApp {
        color: #333333; /* Dark gray font color */
    }
    
    /* Specific styles for Page 3, 4, and 5 */
    .stPage4 h1, .stPage4 h2, .stPage4 h3, .stPage4 p,
    .stPage5 h1, .stPage5 h2, .stPage5 h3, .stPage5 p,
    .stPage6 h1, .stPage6 h2, .stPage6 h3, .stPage6 p {
        color: #333333 !important; /* Dark gray font color */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("🗂️ Page Navigation")
options = st.sidebar.radio("Select a Page",
                           ["Overview",
                            "Data Preprocessing",
                            "Data Visualization",
                            "Models Evaluation",
                            "Predict Attrition",
                            "Upload File for Prediction"])
# Page 1: Overview
if options == "Overview":
    # Description
    st.write("""
    Welcome to the **Employee Attrition Prediction Dashboard**! This interactive dashboard allows you to explore and analyze employee attrition data, 
    compare the performance of different machine learning models and predict attrition for individual employees or based on uploaded datasets.

    ### Key Features:
    - **Dataset Overview**: Explore the dataset, view basic statistics, and check for missing or duplicate data.
    - **Data Visualization**: Visualize key trends and patterns in the dataset using interactive charts.
    - **Model Performance Comparison**: Compare the performance of different machine learning models used for attrition prediction.
    - **Predict Attrition**: Predict whether an employee is likely to leave based on their attributes.
    - **Upload File for Prediction**: Upload a CSV file to predict attrition for multiple employees at once.

    Use the sidebar to navigate between pages and explore the features of this dashboard.
    """)

# Page 2: Data Preprocessing
if options == "Data Preprocessing":
    st.header("🔄 Data Preprocessing")
    st.write("This page provides an overview & preprocessing of the dataset.")

    # Display the first 5 rows of the dataset
    st.subheader("Sample First 5 Rows of the Dataset:")
    st.write(emp_data.head())

    # Display basic info
    st.subheader("Dataset Info:")
    st.write("Numerical Columns:")
    st.write(column_info['Numerical Columns'].dropna().tolist())
    st.write("Categorical Columns:")
    st.write(column_info['Categorical Columns'].dropna().tolist())

    # Display basic statistics
    st.subheader("Dataset Statistics:")
    st.write(emp_data.describe())

    # Display missing data
    st.subheader("Missing Data:")
    st.write(emp_data.isnull().sum())

    # Display duplicate records
    st.subheader("Duplicate Records:")
    st.write(f"Number of duplicate records: {emp_data.duplicated().sum()}")

    # Remove duplicate rows
    st.write(f"Dataset shape after before duplicates: {emp_data.shape}")
    emp_data = emp_data.drop_duplicates()
    st.write(f"Dataset shape after removing duplicates: {emp_data.shape}")

    # Download Processed Dataset
    st.write("Download Processed Dataset:")
    csv = emp_data.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Cleaned Dataset",
                       data=csv,
                       file_name="Processed_HR_Dataset.csv",
                       mime="text/csv")

# Page 3: Data Visualization
elif options == "Data Visualization":
    st.header("📊 Data Visualization")
    st.write("This page provides visualizations of the dataset.")

    # Custom CSS for sidebar filter select boxes
    st.markdown(
        """
        <style>
        /* Sidebar Select Box Options */
        .stSelectbox div[data-baseweb="select"] div {
            color: #333333 !important; /* Dark gray */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add interactive filters in the sidebar
    st.sidebar.subheader("Filters for Data Exploration")

    # Filter by Department
    department_filter = st.sidebar.selectbox(
        "Select Department",
        options=["All"] + emp_data['Departments'].unique().tolist()
    )

    # Filter by Salary Level
    salary_filter = st.sidebar.selectbox(
        "Select Salary Level",
        options=["All"] + emp_data['salary'].unique().tolist()
    )

    # Filter by Attrition Status
    attrition_filter = st.sidebar.selectbox(
        "Select Attrition Status",
        options=["All", "Stayed (0)", "Left (1)"]
    )

    # Apply filters to the dataset
    filtered_data = emp_data.copy()

    if department_filter != "All":
        filtered_data = filtered_data[filtered_data['Departments'] == department_filter]

    if salary_filter != "All":
        filtered_data = filtered_data[filtered_data['salary'] == salary_filter]

    if attrition_filter != "All":
        attrition_value = 0 if attrition_filter == "Stayed (0)" else 1
        filtered_data = filtered_data[filtered_data['left'] == attrition_value]

    # Display filtered dataset
    st.subheader("Filtered Dataset:")
    st.write(filtered_data)

    # Plot Attrition Distribution
    st.subheader("Attrition Distribution:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='left',
                  data=filtered_data,
                  ax=ax, palette='viridis', hue='left', legend=False)
    ax.set_title("Employee Attrition Distribution")
    ax.set_xlabel("Attrition (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Plot Time Spent Distribution
    st.subheader("Time Spent in Company Distribution:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_data['time_spend_company'],
                 kde=True, ax=ax, color='orange', bins=10)
    ax.set_title("Time Spent in Company Distribution")
    ax.set_xlabel("Years Spent in Company")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Plot Department Distribution
    st.subheader("Department Distribution:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='Departments', data=filtered_data, ax=ax,
                  palette='plasma', hue='left', legend=True)
    ax.set_title("Employee Distribution Across Departments")
    ax.set_xlabel("Count")
    ax.set_ylabel("Department")
    st.pyplot(fig)

    # Plot Number of Projects Distribution
    st.subheader("Number of Projects Distribution:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='number_project', data=filtered_data, ax=ax,
                  palette='magma', hue='left', legend=True)
    ax.set_title("Number of Projects Employees Are Involved In")
    ax.set_xlabel("Number of Projects")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Plot Salary Level Distribution
    st.subheader("Salary Level Distribution:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='salary', data=filtered_data, ax=ax,
                  palette='coolwarm', hue='left', legend=True)
    ax.set_title("Employee Salary Level Distribution")
    ax.set_xlabel("Salary Level")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Plot Satisfaction Level Distribution
    st.subheader("Satisfaction Level Distribution:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_data['satisfaction_level'],
                 kde=True, ax=ax, color='green', bins=20)
    ax.set_title("Employee Satisfaction Level Distribution")
    ax.set_xlabel("Satisfaction Level (0 to 1)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Plot Correlation Heatmap
    st.subheader("Correlation Heatmap:")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Encode categorical columns for correlation analysis
    emp_data_encoded = emp_data.copy()
    emp_data_encoded['salary'] = LabelEncoder().fit_transform(
        emp_data_encoded['salary'])
    emp_data_encoded['Departments'] = LabelEncoder().fit_transform(
        emp_data_encoded['Departments'])

    sns.heatmap(emp_data_encoded.corr(), annot=True, fmt=".1f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# Page 4: Model Performance Comparison
elif options == "Models Evaluation":
    st.markdown('<div class="stPage4">', unsafe_allow_html=True)
    st.header("📈 Models Evaluation")
    st.write("This page compares the performance of the four models used for employee attrition prediction.")

    model_performance = {
       "Model": ["Decision Tree", "BalancedRandomForest", "Random Forest", "Gradient Boosting"],
       "Training Accuracy": [f"{100.0:.2f}", f"{99.74:.2f}", f"{100.0:.2f}", f"{98.35:.2f}"],
       "Testing Accuracy": [f"{96.99:.2f}", f"{97.92:.2f}", f"{97.92:.2f}", f"{97.75:.2f}"],
       "Precision (Class 1)": [f"{0.919:.2f}", f"{0.973:.2f}", f"{0.978:.2f}", f"{0.955:.2f}"],
       "Recall (Class 1)": [f"{0.900:.2f}", f"{0.900:.2f}", f"{0.895:.2f}", f"{0.908:.2f}"],
       "F1-Score (Class 1)": [f"{0.909:.2f}", f"{0.935:.2f}", f"{0.935:.2f}", f"{0.931:.2f}"],
       "AUC": [f"{0.942:.2f}", f"{0.981:.2f}", f"{0.974:.2f}", f"{0.981:.2f}"]}

    # Convert to DataFrame for better display
    model_performance_df = pd.DataFrame(model_performance)
    st.subheader("Model Performance Metrics:")
    st.table(model_performance_df)

    # Visualize the metrics using bar charts
    st.subheader("Visual Comparison of Model Performance:")

    # List of metrics to plot
    metrics = ["Training Accuracy", "Testing Accuracy", "Precision (Class 1)", "Recall (Class 1)", "F1-Score (Class 1)", "AUC"]

    # Create a line graph for each metric
    for metric in metrics:
        st.write(f"#### {metric}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="Model", y=metric, data=model_performance_df, marker="o", ax=ax)

        ax.set_title(f"{metric} Comparison")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

# Page 5: Predict Future Attrition
elif options == "Predict Attrition":
    st.markdown('<div class="stPage5">', unsafe_allow_html=True)
    st.header("🔮 Predict Future Employee Attrition")
    st.write("This page allows you to predict attrition based on user selection, based on the best performing model (Gradient Boosting).")

    # Input fields for new data
    st.subheader("Input Employee Data:")

    # Use dropdowns/radio button for categorical inputs
    department = st.selectbox("Department", emp_data['Departments'].unique())
    salary = st.selectbox("Salary Level", emp_data['salary'].unique())
    promotion = st.radio("Promotion in the last 5 Years", ["Yes", "No"])
    work_accident = st.radio("Work Accident", ["Yes", "No"])

    # Use sliders for numerical inputs
    time_spent = st.slider("Total Working Years",
                           min_value=2, max_value=10, value=5)
    monthly_hr = st.slider("Monthly Hour",
                           min_value=96, max_value=310, value=100)
    number_project = st.slider("Total Projects Involved",
                               min_value=2, max_value=7, value=5)
    last_performance = st.slider("Last performance Evaluation (0 to 1)",
                                 min_value=0.36,
                                 max_value=1.0,
                                 value=0.5)
    satistaction = st.slider("Satisfaction level (0 to 1)",
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

    st.markdown('</div>', unsafe_allow_html=True)

# Page 6: Upload File for Prediction
elif options == "Upload File for Prediction":
    st.markdown('<div class="stPage6">', unsafe_allow_html=True)
    st.header("📤 Upload File for Prediction")
    st.write("This page allows you to predict employee attrition based on file upload.")

    # Display the expected dataset structure
    st.subheader("Expected Dataset Structure")
    st.write("Please ensure your dataset has the following columns and data types:")

    # Define the expected dataset structure
    expected_structure = {
        "Column Name": [
            "satisfaction_level", "last_evaluation", "number_project",
            "average_montly_hours", "time_spend_company", "Work_accident",
            "promotion_last_5years", "Departments", "salary"
        ],
        "Data Type": [
            "float (0 to 1)", "float (0 to 1)", "int",
            "int", "int", "int (0 or 1)",
            "int (0 or 1)", "string", "string (low, medium, high)"
        ]
    }

    # Convert to DataFrame for better display
    expected_structure_df = pd.DataFrame(expected_structure)
    st.table(expected_structure_df)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded file
        uploaded_data = pd.read_csv(uploaded_file)

        # Display the first few rows of the uploaded file
        st.subheader("Uploaded File Preview")
        st.write(uploaded_data.head())

        # Check if the uploaded file matches the expected structure
        expected_columns = expected_structure["Column Name"]

        # Normalize column names (strip whitespace and convert to lowercase)
        uploaded_columns = [col.strip().lower() for col in uploaded_data.columns]
        expected_columns_normalized = [col.strip().lower() for col in expected_columns]

        if not all(col in uploaded_columns for col in expected_columns_normalized):
            st.error(f"Error: The uploaded file does not match the expected structure. Please check the column names. Expected: {expected_columns}, Found: {uploaded_data.columns.tolist()}")
        else:
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
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by Nor Hanisah Ahmed Fuad(MRT231005)")
