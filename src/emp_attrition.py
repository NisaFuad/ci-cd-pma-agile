# -*- coding: utf-8 -*-


"""
Import necessary Python libraries, Load the dataset,  Exploratory Data Analysis
"""

# Libray for Data Manipulation
import pandas as pd
import numpy as np

# Library for supervised learning algorithm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
employee_data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

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
    print("-"*140)


# Dropping 4 attributes which doesn't give any meaningful insights in analysis
employee_data = employee_data.drop([
    'Over18', 'EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1)


# Checking the count of target variables
print("\nTarget Variable Info:", employee_data['Attrition'].value_counts())


# Encoding for categorical features for Correlation Analysis

# Replace 'Gender' column values with numerical labels using manual encoding
employee_data["Gender"] = employee_data["Gender"].replace({"Female": 0,
                                                           "Male": 1})

# Perform LabelEncoder for 'Attrition' column
le = LabelEncoder()
employee_data["Attrition"] = le.fit_transform(employee_data['Attrition'])

# Perform OneHotEncoder for multiple categorical columns
encoder = OneHotEncoder()
encoded = encoder.fit_transform(
    employee_data[['Department',
                   'EducationField',
                   'JobRole',
                   'MaritalStatus',
                   'OverTime',
                   'BusinessTravel']])

encoded_df = pd.DataFrame(encoded.toarray(),
                          columns=encoder.get_feature_names_out())
employee_data = pd.concat([employee_data, encoded_df], axis=1)
employee_data = employee_data.drop(
    ['Department',
     'EducationField',
     'JobRole',
     'MaritalStatus',
     'OverTime',
     'BusinessTravel'], axis=1)

# Checking Multicollinearity

# Calculate the correlation matrix
corr_matrix = employee_data.corr()

# Create a mask
high_corr_mask = corr_matrix >= 0.75

# Identify and list the highly correlated features (multicollinearity)
highly_corr_feat = []

for feat in high_corr_mask.columns:
    corr_with = high_corr_mask.index[
        high_corr_mask[feat]].tolist()
    for corr_feat in corr_with:
        if feat != corr_feat and (corr_feat, feat) not in highly_corr_feat:
            highly_corr_feat.append((feat, corr_feat))

# Print the highly correlated features
print("\nHighly correlated features (multicollinearity):")
for feature1, feature2 in highly_corr_feat:
    print(f"{feature1} and {feature2}")

# droping columns which are highly correlated

cols = ["JobLevel", "TotalWorkingYears", "PercentSalaryHike",
        "YearsInCurrentRole", "YearsWithCurrManager"]
employee_data.drop(columns=cols, inplace=True)


# Split the data into Independent (X) and Target Variable (Y)

x = employee_data.drop(['Attrition'], axis=1)
y = employee_data[['Attrition']]

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
