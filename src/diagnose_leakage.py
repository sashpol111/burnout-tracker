"""
diagnose_leakage.py — verify the new target is NOT reconstructible from features.
Run this after any target change to confirm R² << 1.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, BURNOUT_SYMPTOM_COLS

df = load_data()
X, y, feature_cols = preprocess(df)

df_enc = load_data()
df_enc = df_enc.drop(columns=['Timestamp'])
df_enc['GENDER'] = df_enc['GENDER'].map({'Female': 0, 'Male': 1})
df_enc['AGE']    = df_enc['AGE'].map({'Less than 20': 0, '21 to 35': 1,
                                       '36 to 50': 2,   '51 or more': 3})
df_enc = df_enc.apply(pd.to_numeric, errors='coerce').dropna()

burnout_index = df_enc[BURNOUT_SYMPTOM_COLS].sum(axis=1)
feat_df       = df_enc.drop(columns=BURNOUT_SYMPTOM_COLS + ['WORK_LIFE_BALANCE_SCORE'])

X_tr, X_te, y_tr, y_te = train_test_split(feat_df, burnout_index,
                                            test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_tr, y_tr)
r2 = r2_score(y_te, lr.predict(X_te))

print(f"Linear R² of burnout_index ~ wellness features: {r2:.4f}")
print(f"  → {'LEAK DETECTED — check target construction' if r2 > 0.9 else 'OK — target is not reconstructible from features'}")


print(f"\nClass distribution:")
print(y.value_counts().rename({0: 'Low risk', 1: 'High risk'}).to_string())
print(f"Positive rate: {y.mean():.1%}")

threshold = burnout_index.quantile(0.70)
margin    = burnout_index.std() * 0.10
border    = ((burnout_index - threshold).abs() < margin).mean()
print(f"\nBorderline rows (within {margin:.1f} of threshold): {border:.1%}")
print(f"  → {'Low ambiguity — may still be deterministic' if border < 0.05 else 'Good — genuine ambiguity around boundary'}")