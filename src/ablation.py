# ablation.py
# Ablation study varying feature groups to isolate each group's contribution.
# Five conditions tested under identical split, model, and threshold tuning.

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '.')

from data.data_loader import load_data, preprocess
from src.smote import smote
from src.hyperparameter_tuning import grid_search
from src.preprocessing_experiment import find_best_threshold


# train XGBoost on a feature subset, tune threshold on val set, evaluate on test set
def train_and_evaluate(X, y, label, best_alpha, best_lambda):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    X_train_r, y_train_r = smote(X_train_s, y_train.values, random_state=42)

    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=best_alpha, reg_lambda=best_lambda,
        eval_metric='logloss', early_stopping_rounds=15, random_state=42,
    )
    model.fit(X_train_r, y_train_r, eval_set=[(X_val_s, y_val)], verbose=False)

    val_proba         = model.predict_proba(X_val_s)[:, 1]
    threshold, val_f1 = find_best_threshold(y_val, val_proba)

    test_proba = model.predict_proba(X_test_s)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)
    f1         = f1_score(y_test, test_preds, zero_division=0)
    auc        = roc_auc_score(y_test, test_proba)
    acc        = accuracy_score(y_test, test_preds)

    print(f"  {label:<45s} | F1: {f1:.3f} | AUC: {auc:.3f} | "
          f"Acc: {acc:.3f} | feats: {X.shape[1]:2d} | thresh: {threshold:.2f}")
    return f1, auc, X.shape[1]


if __name__ == '__main__':
    df = load_data()
    X, y, feature_cols = preprocess(df, use_domain_cleaning=True)

    print("Running hyperparameter tuning for ablation conditions...")
    best_alpha, best_lambda = grid_search()
    print(f"Using reg_alpha={best_alpha}, reg_lambda={best_lambda}\n")

    print("=" * 75)
    print("ABLATION STUDY — Feature Group Contributions")
    print("=" * 75)
    print(f"  {'Condition':<45s} | F1    | AUC   | Acc   | feats | thresh")
    print("  " + "-" * 70)

    results = {}

    results['All features'] = train_and_evaluate(
        X, y, "1. All features (reference)", best_alpha, best_lambda)

    drop_productivity = ['ACHIEVEMENT', 'SUPPORTING_OTHERS', 'TODO_COMPLETED',
                         'PERSONAL_AWARDS', 'DONATION']
    X_no_prod = X.drop(columns=[c for c in drop_productivity if c in X.columns])
    results['No productivity'] = train_and_evaluate(
        X_no_prod, y, "2. No productivity metrics (drop 5)", best_alpha, best_lambda)

    # DAILY_STRESS, LOST_VACATION, DAILY_SHOUTING already excluded by preprocess()
    health_features = [c for c in [
        'SLEEP_HOURS', 'BMI_RANGE', 'WEEKLY_MEDITATION',
        'DAILY_STEPS', 'FRUITS_VEGGIES', 'TIME_FOR_PASSION',
        'RECOVERY_SCORE', 'HEALTH_HABITS',
    ] if c in X.columns]
    results['Health only'] = train_and_evaluate(
        X[health_features], y, "3. Health/recovery features only", best_alpha, best_lambda)

    X_no_demo = X.drop(columns=[c for c in ['AGE', 'GENDER'] if c in X.columns])
    results['No demographics'] = train_and_evaluate(
        X_no_demo, y, "4. No demographic features", best_alpha, best_lambda)

    social_features = [c for c in [
        'SOCIAL_NETWORK', 'CORE_CIRCLE', 'SUPPORTING_OTHERS',
        'SOCIAL_SUPPORT_SCORE', 'PLACES_VISITED', 'DONATION',
    ] if c in X.columns]
    results['Social only'] = train_and_evaluate(
        X[social_features], y, "5. Social features only", best_alpha, best_lambda)

    print("\n" + "=" * 75)
    print("ABLATION SUMMARY")
    print("=" * 75)
    ref_f1  = results['All features'][0]
    ref_auc = results['All features'][1]
    print(f"  {'Condition':<30s} | F1 drop vs ref | AUC drop vs ref")
    print("  " + "-" * 58)
    for name, (f1, auc, _) in results.items():
        f1_drop  = ref_f1 - f1
        auc_drop =