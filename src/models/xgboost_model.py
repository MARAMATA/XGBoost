import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.best_model = None
    
    def train(self, X_train, y_train):
        params = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200]
        }
        
        grid_search = GridSearchCV(self.model, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
    
    def evaluate(self, X_test, y_test):
        y_pred = self.best_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        auc_roc = roc_auc_score(y_test, self.best_model.predict_proba(X_test)[:, 1])
        print(f"AUC-ROC: {auc_roc:.4f}")
    
    def save_model(self, path='models/xgb_model.pkl'):
        joblib.dump(self.best_model, path)