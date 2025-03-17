from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
from flask_cors import CORS

from recommendation import label_encoder

app = Flask(__name__)
CORS(app)

with open("predictionModel.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("label_encoder.pkl", "rb") as enc_file:
    medical_condition_encoder = pickle.load(enc_file)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Receive JSON data from request
        df = pd.DataFrame([data])  # Convert to DataFrame
        print(df)
        # label_encoder=LabelEncoder()
        # Ensure columns match the model input (feature columns)
        feature_columns = ['Age','Gender','Blood Type','Medication','Test Results']
        df = df[feature_columns]
        df['Gender']=medical_condition_encoder.fit_transform(df['Gender'])
        df['Blood Type'] = medical_condition_encoder.fit_transform(df['Blood Type'])
        df['Medication'] = medical_condition_encoder.fit_transform(df['Medication'])
        df['Test Results'] = medical_condition_encoder.fit_transform(df['Test Results'])

        prediction = model.predict(df)
        print(prediction)
        predicted_condition=medical_condition_encoder.inverse_transform([prediction[0]])[0]

        # Return the prediction
        return jsonify({"predicted_medical_condition": predicted_condition})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True,port=8000)
