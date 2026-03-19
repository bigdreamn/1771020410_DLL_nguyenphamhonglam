import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score
import shap

def train_supervised_models(df, target_col='left', test_size=0.2, random_state=42):
    """
    Train and evaluate XGBoost and Random Forest models.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                        random_state=random_state, use_label_encoder=False, 
                        eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    results = {
        'xgb': {'model': xgb, 'pred': y_pred_xgb, 'prob': y_prob_xgb},
        'rf': {'model': rf, 'pred': y_pred_rf, 'prob': y_prob_rf},
        'y_test': y_test,
        'X_test': X_test
    }
    
    return results

def get_metrics(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    return pr_auc

def explain_model(model, X_test):
    """
    Explain model using SHAP.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values
