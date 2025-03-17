from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS

from recommendation import label_encoder

app = Flask(__name__)
CORS(app)

with open("hc_doctorPredictionModel.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("hc_label_encoder.pkl", "rb") as enc_file:
    medical_condition_encoder = pickle.load(enc_file)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Receive JSON data from request
        # label_encoder=LabelEncoder()
        input_data=np.array([
            medical_condition_encoder['Gender'].transform([data['Gender']])[0],
            medical_condition_encoder['Blood Type'].transform([data['Blood Type']])[0],
            medical_condition_encoder['Medical Condition'].transform([data['Medical Condition']])[0],
        ]).reshape(1,-1)
        df=pd.DataFrame(input_data,columns=['Gender','Blood Type','Medical Condition'])
        print(df)
        prediction=model.predict(input_data)
        print("prediction",prediction[0])
        doctor=medical_condition_encoder['Doctor'].inverse_transform(prediction)[0]
        print("doctor",doctor)

        # doctors_probabilities=medical_condition_encoder.inverse_transform(prediction)[0]
        # doctor_labels=model_classes
        # print("doctors",doctors)

        return jsonify({"doctor":doctor})
    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(debug=True, port=8000)