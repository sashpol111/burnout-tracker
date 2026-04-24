import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, split_and_scale

def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, proba)
    print(f"{split_name} — Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    return acc, f1, auc

if __name__ == '__main__':
    df = load_data()
    X, y, feature_cols = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=20,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )

    print("\n=== XGBoost Results ===")
    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_val, y_val, "Val")
    evaluate(model, X_test, y_test, "Test")

    # Save model and scaler
    joblib.dump(model, 'models/xgboost_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    print("\nModel saved to models/")