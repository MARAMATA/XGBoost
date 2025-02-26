from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import numpy as np
from src.controllers.train_controller import train_model
from src.controllers.predict_controller import predict
from src.controllers.evaluation_controller import evaluate_model

app = Flask(__name__)
Swagger(app, template_file='swagger.yml')

@app.route('/', methods=['GET'])
def home():
    """
    Home Endpoint
    ---
    responses:
      200:
        description: API is running successfully
    """
    return jsonify({"message": "Bienvenue sur l'API de détection d'anomalies avec XGBoost"}), 200

@app.route('/train', methods=['POST'])
def train():
    """
    Train the XGBoost Model
    ---
    responses:
      200:
        description: Model trained and saved successfully
    """
    try:
        train_model()
        return jsonify({"message": "Model trained and saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Predict Anomalies using the trained model
    ---
    parameters:
      - name: data
        in: body
        required: true
        schema:
          type: object
          properties:
            data:
              type: array
              items:
                type: number
    responses:
      200:
        description: Prediction result
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request. Please send JSON data."}), 400
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "Invalid request. Please provide 'data' in JSON format."}), 400
        
        model = joblib.load('models/xgb_model.pkl')
        scaler_data = joblib.load('models/scaler.pkl')
        scaler = scaler_data['scaler']
        feature_names = scaler_data['feature_names']
        
        input_data = np.array(data["data"]).reshape(1, -1)
        if input_data.shape[1] != len(feature_names):
            return jsonify({"error": f"Input has {input_data.shape[1]} features, but model expects {len(feature_names)}."}), 400
        
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        return jsonify({"prediction": prediction.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['GET'])
def evaluate():
    """
    Evaluate the model's performance
    ---
    responses:
      200:
        description: Model evaluation scores
    """
    scores = evaluate_model()
    return jsonify(scores), 200

# ⬇️ Ajoute cette ligne pour que Vercel puisse exécuter l'application
def handler(event, context):
    return app
