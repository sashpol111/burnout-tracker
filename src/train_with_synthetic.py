import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, label):
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
    print(f"{label:40s} | Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    return acc, f1, auc

if __name__ == '__main__':
    # load real data
    df_real = load_data()
    X_real, y_real, feature_cols = preprocess(df_real)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"\n{'Model':40s} | Acc   | F1    | AUC")
    print("-" * 65)
    
    # baseline: real data only
    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, "Real data only")
    
    # synthetic data
    df_syn = pd.read_csv('data/synthetic_burnout_data.csv')
    df_syn = df_syn[feature_cols + ['BURNOUT_RISK']]
    df_syn = df_syn.apply(pd.to_numeric, errors='coerce').dropna()
    
    X_syn = df_syn[feature_cols]
    y_syn = df_syn['BURNOUT_RISK']
    
    # augmented: real + synthetic training data
    X_train_aug = pd.concat([X_train, X_syn], ignore_index=True)
    y_train_aug = pd.concat([y_train, y_syn], ignore_index=True)
    
    train_and_evaluate(X_train_aug, y_train_aug, X_val, y_val, X_test, y_test, "Real + synthetic data")
    
    print(f"\nSynthetic entries added: {len(X_syn)}")
    print(f"Training set size: {len(X_train)} -> {len(X_train_aug)}")