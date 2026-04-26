import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess

def train_and_evaluate(X, y, label):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss',
        early_stopping_rounds=15, random_state=42
    )
    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    
    print(f"{label:45s} | Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    return acc, f1, auc

if __name__ == '__main__':
    df = load_data()
    X, y, feature_cols = preprocess(df)
    
    print(f"\n{'Feature Set':45s} | Acc   | F1    | AUC")
    print("-" * 75)
    
    # experiment 1: All features
    train_and_evaluate(X, y, "All 22 features")
    
    # experiment 2: Remove high-correlation productivity metrics
    drop_productivity = ['ACHIEVEMENT', 'SUPPORTING_OTHERS', 'TODO_COMPLETED', 
                         'PERSONAL_AWARDS', 'DONATION']
    X_no_prod = X.drop(columns=[c for c in drop_productivity if c in X.columns])
    train_and_evaluate(X_no_prod, y, "No productivity metrics (drop 5)")
    
    # experiment 3: Only stress/health signals
    stress_features = ['DAILY_STRESS', 'SLEEP_HOURS', 'LOST_VACATION', 
                       'DAILY_SHOUTING', 'BMI_RANGE', 'WEEKLY_MEDITATION',
                       'DAILY_STEPS', 'FRUITS_VEGGIES']
    X_stress = X[stress_features]
    train_and_evaluate(X_stress, y, "Stress/health signals only (8 features)")
    
    # experiment 4: Remove demographic features
    X_no_demo = X.drop(columns=['AGE', 'GENDER'])
    train_and_evaluate(X_no_demo, y, "No demographic features")
    
    # experiment 5: Top 10 most important features only
    top10 = ['TODO_COMPLETED', 'PLACES_VISITED', 'ACHIEVEMENT', 'SUPPORTING_OTHERS',
             'SUFFICIENT_INCOME', 'FLOW', 'TIME_FOR_PASSION', 'CORE_CIRCLE',
             'DAILY_STRESS', 'SLEEP_HOURS']
    X_top10 = X[top10]
    train_and_evaluate(X_top10, y, "Top 10 important features only")