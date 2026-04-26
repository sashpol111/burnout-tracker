import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BURNOUT_SYMPTOM_COLS = ['DAILY_STRESS', 'DAILY_SHOUTING', 'LOST_VACATION']
TARGET_ADJACENT_COLS = BURNOUT_SYMPTOM_COLS + ['WORK_LIFE_BALANCE_SCORE', 'BURNOUT_RISK']

# 0 is an impossible response for these
LIKERT_1_10 = [
    'PERSONAL_AWARDS', 'WEEKLY_MEDITATION', 'SLEEP_HOURS',
    'SOCIAL_NETWORK', 'TODO_COMPLETED', 'CORE_CIRCLE', 'SUPPORTING_OTHERS',
    'SUFFICIENT_INCOME', 'BMI_RANGE',
]
LIKERT_0_10 = [
    'TIME_FOR_PASSION', 'ACHIEVEMENT', 'DONATION', 'FLOW',
    'LIVE_VISION', 'PLACES_VISITED', 'FRUITS_VEGGIES',
]
MAX_OOR_RATE = 0.05


def load_data(path='data/unified_dataset.csv'):
    return pd.read_csv(path)

def _apply_domain_cleaning(df):
    # skips columns where OOR rate > MAX_OOR_RATE 
    total_fixed, clamped_cols, skipped_cols = 0, [], []

    for col, lo, hi in (
        [(c, 1, 10) for c in LIKERT_1_10] +
        [(c, 0, 10) for c in LIKERT_0_10]
    ):
        if col not in df.columns:
            continue
        oor_mask = (df[col] < lo) | (df[col] > hi)
        oor_rate = oor_mask.mean()
        if oor_rate == 0:
            continue
        if oor_rate > MAX_OOR_RATE:
            skipped_cols.append(f"{col} ({oor_rate:.1%} OOR)")
            continue
        n = int(oor_mask.sum())
        df[col] = df[col].clip(lo, hi)
        total_fixed += n
        clamped_cols.append(f"{col} ({n} values → [{lo},{hi}])")

    print(f"[domain cleaning] Fixed {total_fixed} errors: "
          f"{', '.join(clamped_cols) or 'none'}")
    if skipped_cols:
        print(f"[domain cleaning] Skipped: {'; '.join(skipped_cols)}")


def preprocess(df, use_domain_cleaning=False):
    # detect source: kaggle raw has Timestamp, string GENDER/AGE, symptom cols
    is_kaggle_raw = (
        'Timestamp' in df.columns or
        ('GENDER' in df.columns and df['GENDER'].dtype == object) or
        ('BURNOUT_RISK' not in df.columns and
         all(c in df.columns for c in BURNOUT_SYMPTOM_COLS))
    )

    if is_kaggle_raw:
        df = df.drop(columns=['Timestamp'], errors='ignore')

        if 'GENDER' in df.columns and df['GENDER'].dtype == object:
            df['GENDER'] = df['GENDER'].map({'Female': 0, 'Male': 1})

        if 'AGE' in df.columns and df['AGE'].dtype == object:
            df['AGE'] = df['AGE'].map({'Less than 20': 0, '21 to 35': 1,
                                       '36 to 50': 2,    '51 or more': 3})

        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if use_domain_cleaning:
            _apply_domain_cleaning(df)

        burnout_index      = df[BURNOUT_SYMPTOM_COLS].sum(axis=1)
        threshold          = burnout_index.quantile(0.70)
        df['BURNOUT_RISK'] = (burnout_index >= threshold).astype(int)
        pos_rate           = df['BURNOUT_RISK'].mean()
        print(f"[target] threshold={threshold:.1f} | positive rate={pos_rate:.1%} | "
              f"class ratio 1:{(1-pos_rate)/pos_rate:.1f}")

    else:
        # unified dataset: drop text col before coerce
        df = df.drop(columns=['source'], errors='ignore')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if use_domain_cleaning:
            _apply_domain_cleaning(df)

        pos_rate = df['BURNOUT_RISK'].mean()
        print(f"[unified dataset] {len(df)} rows | positive rate={pos_rate:.1%} | "
              f"class ratio 1:{(1-pos_rate)/pos_rate:.1f}")

    df['RECOVERY_SCORE']       = (df['SLEEP_HOURS']
                                   + df['TIME_FOR_PASSION']
                                   + df['WEEKLY_MEDITATION'])
    df['SOCIAL_SUPPORT_SCORE'] = df['SOCIAL_NETWORK'] + df['CORE_CIRCLE']
    df['LIFESTYLE_SCORE']      = (df['FLOW'] + df['ACHIEVEMENT']
                                   + df['LIVE_VISION'] + df['TIME_FOR_PASSION'])
    df['HEALTH_HABITS']        = (df['FRUITS_VEGGIES']
                                   + df['SLEEP_HOURS']
                                   + df['TODO_COMPLETED'])

    feature_cols = [c for c in df.columns if c not in TARGET_ADJACENT_COLS]
    return df[feature_cols], df['BURNOUT_RISK'], feature_cols


def split_and_scale(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


if __name__ == '__main__':
    df = load_data()
    X, y, cols = preprocess(df, use_domain_cleaning=True)
    split_and_scale(X, y)
    print(f"Features ({len(cols)}): {cols}")

    if __import__('os').path.exists('data/unified_dataset.csv'):
        df2 = load_data()
        X2, y2, cols2 = preprocess(df2, use_domain_cleaning=True)
        split_and_scale(X2, y2)
    else:
        print("unified_dataset.csv not found — run data/data_pipeline.py first")