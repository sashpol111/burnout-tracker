import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, split_and_scale

def evaluate(model, X, y):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return accuracy_score(y, preds), f1_score(y, preds), roc_auc_score(y, proba)

if __name__ == '__main__':
    df = load_data()
    X, y, feature_cols = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)

    configs = [
        {
            'name': 'Config 1: Deep trees, high LR',
            'params': dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                          early_stopping_rounds=20, random_state=42)
        },
        {
            'name': 'Config 2: Regularized (final model)',
            'params': dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                          subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                          reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss',
                          early_stopping_rounds=15, random_state=42)
        },
        {
            'name': 'Config 3: Shallow trees, low LR',
            'params': dict(n_estimators=200, max_depth=3, learning_rate=0.01,
                          subsample=0.6, colsample_bytree=0.6, min_child_weight=10,
                          reg_alpha=0.5, reg_lambda=2.0, eval_metric='logloss',
                          early_stopping_rounds=15, random_state=42)
        }
    ]

    print(f"\n{'Config':35s} | Val F1 | Val AUC | Test F1 | Test AUC")
    print("-" * 80)

    for config in configs:
        model = XGBClassifier(**config['params'])
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        _, val_f1, val_auc = evaluate(model, X_val, y_val)
        _, test_f1, test_auc = evaluate(model, X_test, y_test)
        
        print(f"{config['name']:35s} | {val_f1:.3f}  | {val_auc:.3f}   | {test_f1:.3f}   | {test_auc:.3f}")