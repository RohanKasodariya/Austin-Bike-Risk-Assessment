# -*- coding: utf-8 -*-
"""Module5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FElxGuJxMajcM4smSKtpM1BSblXKwHPb
"""

import pandas as pd
import numpy as np
!pip install scikit-learn==1.3.0

df = pd.read_csv('/content/bike_crash-B-PF307G4M .csv')

"""# Data Preprocessing"""

df.head()

df.drop(columns = ['$1000 Damage to Any One Person\'s Property','Active School Zone Flag','At Intersection Flag','Average Daily Traffic Amount','Construction Zone Flag','Crash Total Injury Count','Crash Year','Intersection Related','Traffic Control Type'],inplace =True)

df.head()

df.describe()

df['Crash Severity'].value_counts()

# converting 'crash severity' to high, medium, and low risk categories

# Define the mapping from original crash severities to risk levels
risk_mapping = {
    'Non-Incapacitating Injury': 'Medium',
    'Possible Injury': 'Low',
    'Incapacitating Injury': 'High',
    'Not Injured': 'Low',
    'Killed': 'High'
}

df['Crash Severity'] = df['Crash Severity'].map(risk_mapping)

from sklearn.utils import resample

# Find the number of samples in the majority class
majority_class_size = df['Crash Severity'].value_counts().max()

# Upsample the minority classes
df_upsampled_list = []
for severity in df['Crash Severity'].unique():
    df_severity = df[df['Crash Severity'] == severity]
    df_severity_upsampled = resample(df_severity,
                                     replace=True,
                                     n_samples=majority_class_size,
                                     random_state=123)
    df_upsampled_list.append(df_severity_upsampled)

# Concatenate the upsampled minority classes with the rest of the data
df_balanced = pd.concat(df_upsampled_list)

# Display new class counts
print(df_balanced['Crash Severity'].value_counts())

df_balanced.info()

# Standardize "Crash Time" to a four-digit HHMM format with leading zeros
df_balanced['Crash Time'] = df_balanced['Crash Time'].apply(lambda x: f"{x:04d}")
# Check the first few standardized entries
df_balanced['Crash Time'].head()

df_balanced['Crash Time'] = df_balanced['Crash Time'].astype(int)

skewness = df_balanced['Speed Limit'].skew()

print(f"Skewness of the 'Speed Limit' column: {skewness}")

# Calculate the median speed limit, excluding the -1 values
median_speed_limit = df_balanced[df_balanced['Speed Limit'] > 0]['Speed Limit'].median()

# Replace -1 values with the median
df_balanced['Speed Limit'] = df_balanced['Speed Limit'].replace(-1, median_speed_limit)

# Mapping 'Surface condition' into smaller cateogries
surface_condition_mapping = {
    'Dry': 'Dry',
    'Wet': 'Wet',
    'Unknown': 'Other',
    'Other (Explain In Narrative)': 'Other',
    'Ice': 'Other',
    'Sand, Mud, Dirt': 'Other',
    'Standing Water': 'Other'
}

# Apply the mapping to the 'Surface Condition' column
df_balanced['Surface Condition'] = df_balanced['Surface Condition'].map(surface_condition_mapping)

df_balanced['Person Helmet'].value_counts()

df_balanced.info()

# Mapping 'Person helmet' into smaller cateogries
helmet_mapping_conservative = {
    'Not Worn': 'Not Worn',
    'Unknown If Worn': 'Not Worn',
    'Worn, Not Damaged': 'Worn',
    'Worn, Unk Damage': 'Worn',
    'Worn, Damaged': 'Worn'
}

# Apply the mapping to simplify the "Person Helmet" column
df_balanced['Person Helmet'] = df_balanced['Person Helmet'].map(helmet_mapping_conservative)

df_balanced.head()

"""# EDA"""

df['Time of Day'] = df['Crash Time'].apply(lambda x: 'Day' if 600 <= x <= 1800 else 'Night')

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'Crash Severity' has been mapped to 'High', 'Medium', 'Low' categories
sns.countplot(x='Time of Day', hue='Crash Severity', data=df, order=['Day', 'Night'], hue_order=['High', 'Medium', 'Low'])
plt.title('Crash Severity by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Number of Crashes')
plt.show()

# Example for visualizing crash severity by another variable (helmet usage or surface condition)
variable = 'Person Helmet'  # or 'Surface Condition Mapped' for Hypothesis 3
sns.countplot(x=variable, hue='Crash Severity', data=df, hue_order=['High', 'Medium', 'Low'])
plt.title(f'Crash Severity by {variable}')
plt.xlabel(variable)
plt.ylabel('Number of Crashes')
plt.legend(title='Crash Severity')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Surface Condition', hue='Crash Severity', data=df, palette='viridis')
plt.title('Crash Severity by Road Surface Condition')
plt.xlabel('Road Surface Condition')
plt.ylabel('Number of Crashes')
plt.legend(title='Crash Severity', loc='upper right')
plt.tight_layout()

# Show the plot
plt.show()

"""# Decision Tree Model"""

df.drop('Time of Day',inplace = True, axis =1)

df_balanced['Person Helmet'].value_counts()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Separate features and target variable
X = df_balanced.drop('Crash Severity', axis=1)
y = df_balanced['Crash Severity']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ColumnTransformer
transformer = ColumnTransformer(transformers=[
    ('tnf1', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), ['Day of Week', 'Roadway Part', 'Surface Condition', 'Person Helmet']),
    ('passthrough', 'passthrough', ['Speed Limit', 'Crash Time'])
], remainder='passthrough')

# Fit the transformer on the training data
transformer.fit(X_train)

# Transform the training and testing data
X_train_transformed = transformer.transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Create the decision tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the decision tree on the transformed data
model.fit(X_train_transformed, y_train)

# 3. Model Evaluation
# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test_transformed)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))

import joblib

# Save the transformer and model
joblib.dump(transformer, 'transformer.joblib')
joblib.dump(model, 'decision_tree_model.joblib')

import numpy as np
import matplotlib.pyplot as plt

# Function to get feature names from ColumnTransformer
def get_feature_names(column_transformer):
    output_features = []

    # Loop over all transformers in the ColumnTransformer
    for name, estimator, columns in column_transformer.transformers_:
        if name == 'passthrough':
            # If the transformer is 'passthrough', the feature names come directly from the columns
            output_features.extend(columns)
        elif hasattr(estimator, 'get_feature_names_out'):
            # If the transformer has a 'get_feature_names_out' method, use it
            output_features.extend(estimator.get_feature_names_out(columns))
        else:
            # Otherwise, we assume the feature names are the same as the columns
            output_features.extend(columns)

    return output_features

# Now get the feature names
transformed_feature_names = get_feature_names(transformer)

# Ensure the length of 'transformed_feature_names' matches 'model.feature_importances_'
assert len(transformed_feature_names) == len(model.feature_importances_)

# Now create the DataFrame using the transformed feature names
feature_importance_df = pd.DataFrame({
    'Feature': transformed_feature_names,
    'Importance': model.feature_importances_
})

# Sort the DataFrame by importance
feature_importance_df.sort_values(by='Importance', ascending=True, inplace=True)

# Plot using Matplotlib
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

