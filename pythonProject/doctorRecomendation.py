import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf

df=pd.read_csv('healthcare_dataset.csv')
# print(df)
df=df.head(400)
label_encoder=LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Blood Type'] = label_encoder.fit_transform(df['Blood Type'])
df['Medical Condition'] = label_encoder.fit_transform(df['Medical Condition'])
df['Doctor'] = label_encoder.fit_transform(df['Doctor'].str.upper())
df['Hospital'] = label_encoder.fit_transform(df['Hospital'])
df['Insurance Provider'] = label_encoder.fit_transform(df['Insurance Provider'])
df['Admission Type'] = label_encoder.fit_transform(df['Admission Type'])
df['Medication'] = label_encoder.fit_transform(df['Medication'])
df['Test Results'] = label_encoder.fit_transform(df['Test Results'])

df = df.dropna()

X = df.drop(columns=['Name','Date of Admission','Discharge Date','Gender','Hospital','Insurance Provider','Admission Type','Medication','Billing Amount','Room Number','Admission Type','Doctor'])
y = df['Doctor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")