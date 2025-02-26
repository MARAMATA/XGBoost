import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.data_preprocessing import DataPreprocessor

def evaluate_model():
    try:
        model = joblib.load('models/xgb_model.pkl')
        scaler_data = joblib.load('models/scaler.pkl')
        scaler = scaler_data['scaler']
        
        preprocessor = DataPreprocessor("data/data-20221207.csv")
        preprocessor.load_data()
        preprocessor.clean_data()
        preprocessor.detect_anomalies()
        _, X_test, _, y_test = preprocessor.split_data()
        
        y_pred = model.predict(X_test)
        
        scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        return scores
    except Exception as e:
        return {"error": str(e)}