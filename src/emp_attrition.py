# -*- coding: utf-8 -*-


"""
Import necessary Python libraries, Load the dataset,  Exploratory Data Analysis
"""

# Libray for Data Manipulation
import pandas as pd
import numpy as np

# Library for supervised learning algorithm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score

# Import Joblib & Pickle
import joblib

# Library to overcome Warnings.
import warnings
warnings.filterwarnings('ignore')

# Load dataset
employee_data = pd.read_csv("data/HR_Dataset.csv")

# Print the first five rows
print("\nFirst 5 row of dataset:\n", employee_data.head())

# Shape of the dataset
print("\nInitial dataset shape:", employee_data.shape)

# Basic Info
print("\nStatistic of dataset:\n")
print(employee_data.info())

# Checking for missing data
print("\nMissing Data:\n", employee_data.isnull().sum())

# Checking for duplicate records
print("\nDuplicates in Dataset: ", employee_data.duplicated().sum())

# Remove duplicate rows
employee_data = employee_data.drop_duplicates()
print(f"Dataset shape after removing duplicates: {employee_data.shape}")

# Save the processed dataset
employee_data.to_csv('data/processed_df.csv', index=False)
print("Processed dataset saved to data/processed_df.csv")

# Identify the data types of columns
column_data_types = employee_data.dtypes

# Count the numerical and categorical columns
numerical_count = 0
categorical_count = 0

for column_name, data_type in column_data_types.items():
    if np.issubdtype(data_type, np.number):
        numerical_count += 1
    else:
        categorical_count += 1

# Print the counts
print(f"\nThere are {numerical_count} Numerical Columns in dataset")
print(f"There are {categorical_count} Categorical Columns in dataset\n")

# Random sample of dataset with only numerical feature
employee_data.select_dtypes(np.number).sample(5)

# Random sample of dataset with only categorical feature
employee_data.select_dtypes(include=['object']).sample(5)

# Statistics of the numerical features
employee_data.describe()

# Checking count of cardinality/unique valus of Numerical Attributes
num_cols = employee_data.select_dtypes(include=['integer']).columns
cardinality = {}
for column in num_cols:
    cardinality[column] = len(employee_data[column].unique())
    print('Cardinality of', column, ':', cardinality[column])
    print("-"*140)


# Descriptive Analysis on Categorical Attributes
employee_data.describe(include='object').T

# Checking Unique Values of Categorical Attributes
cat_cols = employee_data.select_dtypes(include=['object']).columns

for column in cat_cols:
    print('Unique values of ', column, set(employee_data[column]))
    print("-"*50)

# Checking the count of target variables
print("\nTarget Variable Info:", employee_data['left'].value_counts())

# creating labelEncoder
le = LabelEncoder()

# Converting string labels into numbers.
employee_data['salary'] = le.fit_transform(employee_data['salary'])
employee_data['Departments'] = le.fit_transform(employee_data['Departments'])

# Split the data into Independent (X) and Target Variable (Y)

x = employee_data.drop(['left'], axis=1)
y = employee_data[['left']]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Print the shape of the training and testing independent variables
print('\nUpdated Dataset Shape:', x.shape)
print('Training Dataset:', x_train.shape)
print('Testing Dataset:', x_test.shape)


# Baseline
training_score = []
testing_score = []
precission = []
recall = []
Roc_Auc_score = []
f1_score_ = []
kappa_score = []
G_Mean = []


# Model Building

def model_prediction_unscaled(model):
    model.fit(x_train, y_train)
    x_train_pred = model.predict(x_train)
    x_test_pred = model.predict(x_test)
    y_test_prob = model.predict_proba(x_test)[:, 1]

    # Save the model using pickle
    model_name = model.__class__.__name__
    filename = f"{model_name}.pkl"
    joblib.dump(model, filename)

    a = accuracy_score(y_train, x_train_pred)*100
    b = accuracy_score(y_test, x_test_pred)*100
    c = precision_score(y_test, x_test_pred)
    d = recall_score(y_test, x_test_pred)
    e = roc_auc_score(y_test, y_test_prob)
    f = f1_score(y_test, x_test_pred)

    training_score.append(a)
    testing_score.append(b)
    precission.append(c)
    recall.append(d)
    Roc_Auc_score.append(e)
    f1_score_.append(f)

    print("\n-----------------------------------------------------")
    print(f"Accuracy_Score of {model} model on Training Data is:", a)
    print(f"Accuracy_Score of {model} model on Testing Data is:", b)
    print(f"Precision Score of {model} model is:", c)
    print(f"Recall Score of {model} model is:", d)
    print(f"AUC Score of {model} model is:", e)
    print(f"F1 Score of {model} model is:", f)

    print("\n-----------------------------------------------------")
    print(f"Classification Report of {model} model is:")
    print(classification_report(y_test, model.predict(x_test)))


# Decision Tree
model_prediction_unscaled(DecisionTreeClassifier())

# Random Forest
model_prediction_unscaled(RandomForestClassifier())

# Gradient Boosting
model_prediction_unscaled(GradientBoostingClassifier())
