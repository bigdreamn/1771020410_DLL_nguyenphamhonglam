import pandas as pd
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc

def run_semi_supervised_experiment(df, target_col='left', p_labels=[0.05, 0.1, 0.2], random_state=42):
    """
    Run semi-supervised experiment with different percentages of labeled data.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into a fixed test set for evaluation
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, 
                                                                  random_state=random_state, stratify=y)
    
    results = {}
    
    for p in p_labels:
        # Create a copy of training labels and mask (1 - p)% as -1 (unlabeled)
        y_train_masked = y_train_full.copy()
        rng = np.random.RandomState(random_state)
        random_unlabeled_indices = rng.rand(len(y_train_full)) > p
        y_train_masked[random_unlabeled_indices] = -1
        
        # Supervised-only (on the p% labeled subset)
        idx_labeled = y_train_masked != -1
        clf_supervised = XGBClassifier(n_estimators=50, random_state=random_state, eval_metric='logloss')
        clf_supervised.fit(X_train_full[idx_labeled], y_train_full[idx_labeled])
        y_prob_sup = clf_supervised.predict_proba(X_test)[:, 1]
        
        # Semi-supervised (Self-training)
        base_clf = XGBClassifier(n_estimators=50, random_state=random_state, eval_metric='logloss')
        self_training_clf = SelfTrainingClassifier(base_clf, threshold=0.75, max_iter=10)
        self_training_clf.fit(X_train_full, y_train_masked)
        y_prob_semi = self_training_clf.predict_proba(X_test)[:, 1]
        
        # Calculate PR-AUC
        precision_sup, recall_sup, _ = precision_recall_curve(y_test, y_prob_sup)
        pr_auc_sup = auc(recall_sup, precision_sup)
        
        precision_semi, recall_semi, _ = precision_recall_curve(y_test, y_prob_semi)
        pr_auc_semi = auc(recall_semi, precision_semi)
        
        results[p] = {
            'supervised_pr_auc': pr_auc_sup,
            'semi_supervised_pr_auc': pr_auc_semi
        }
        
    return results
