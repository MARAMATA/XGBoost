import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
    
    def clean_data(self):
        cols_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Label']
        self.df.drop(columns=[col for col in cols_to_drop if col in self.df.columns], errors='ignore', inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
    
    def detect_anomalies(self):
        df_cleaned = self.df.apply(pd.to_numeric, errors='coerce')
        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        df_cleaned['anomaly'] = iso_forest.fit_predict(df_cleaned)
        df_cleaned['anomaly'].replace({-1: 0, 1: 1}, inplace=True)
        self.df = df_cleaned
    
    def split_data(self):
        X = self.df.drop(columns=['anomaly'])
        y = self.df['anomaly']
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def save_scaler(self, path='models/scaler.pkl'):
        joblib.dump({'scaler': self.scaler, 'feature_names': self.feature_names}, path)