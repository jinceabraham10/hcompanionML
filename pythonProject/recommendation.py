import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('healthcare_dataset.csv')

# Keep only the first 400 records
df = df.head(80000)
print(df['Doctor'].nunique())
# Apply Label Encoding
label_encoder = LabelEncoder()
for col in ['Gender', 'Blood Type', 'Medical Condition', 'Hospital', 'Insurance Provider', 'Admission Type', 'Medication', 'Test Results']:
    df[col] = label_encoder.fit_transform(df[col])

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

# Drop unnecessary columns
# df = df.drop(columns=['Name', 'Date of Admission', 'Discharge Date','Insurance Provider', 'Doctor'])
df=df[['Age','Gender','Blood Type','Medical Condition','Medication','Test Results']]

# Remove NaN values
df = df.dropna()

# Define Features (X) and Target (y)
X = df.drop(columns=['Medical Condition'])  # Features
y = df['Medical Condition']  # Target

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions
y_pred = rf_model.predict(X_test)

# Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

with open('predictionModel.pkl',"wb") as model_file:
    pickle.dump(rf_model,model_file)

print("model saved")
