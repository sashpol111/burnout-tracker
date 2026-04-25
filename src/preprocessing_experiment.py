import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, '.')

from src.data_loader import load_data, preprocess, split_and_scale


def evaluate(model, X, y, name):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, proba)
    print(f"{name:25s} | Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    return acc, f1, auc


if __name__ == '__main__':
    df = load_data()

    print("\n=== Preprocessing Experiment ===")
    print(f"{'Model':25s} | Acc   | F1    | AUC")
    print("-" * 60)

    # -----------------------------
    # BASELINE (no advanced preprocessing)
    # -----------------------------
    X_base, y, feature_cols = preprocess(df.copy(), use_clipping=False)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X_base, y)

    model_base = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='logloss', early_stopping_rounds=15,
        random_state=42
    )

    model_base.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    evaluate(model_base, X_test, y_test, "Baseline")

    # -----------------------------
    # IMPROVED (class imbalance + outlier handling)
    # -----------------------------
    X_clean, y, feature_cols = preprocess(df.copy(), use_clipping=True)

    # IMPORTANT: use SAME split indices
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X_clean, y)

    # Compute class imbalance weight
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model_improved = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='logloss', early_stopping_rounds=15,
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )

    model_improved.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    evaluate(model_improved, X_test, y_test, "Improved (+imbalance +clipping)")