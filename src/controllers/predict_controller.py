import joblib
import numpy as np

def predict(input_data):
    model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return prediction.tolist()