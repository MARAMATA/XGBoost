from src.models.data_preprocessing import DataPreprocessor
from src.models.xgboost_model import XGBoostModel


def train_model():
    preprocessor = DataPreprocessor("data/data-20221207.csv")
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.detect_anomalies()
    X_train, X_test, y_train, y_test = preprocessor.split_data()
    preprocessor.save_scaler()
    
    model = XGBoostModel()
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save_model()