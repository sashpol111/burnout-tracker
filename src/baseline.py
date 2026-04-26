import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, split_and_scale

def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, proba)
    print(f"{split_name} — Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

if __name__ == '__main__':
    df = load_data()
    X, y, feature_cols = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)

    # majority class baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    print("=== Baseline (majority class) ===")
    evaluate(dummy, X_train, y_train, "Train")
    evaluate(dummy, X_val, y_val, "Val")
    evaluate(dummy, X_test, y_test, "Test")