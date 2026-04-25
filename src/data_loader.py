import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BURNOUT_SYMPTOM_COLS = ['DAILY_STRESS', 'DAILY_SHOUTING', 'LOST_VACATION']
TARGET_ADJACENT_COLS = BURNOUT_SYMPTOM_COLS + ['WORK_LIFE_BALANCE_SCORE', 'BURNOUT_RISK']

# Features with valid range [1, 10] — 0 is an impossible response
LIKERT_1_10 = [
    'PERSONAL_AWARDS', 'WEEKLY_MEDITATION', 'SLEEP_HOURS',
    'SOCIAL_NETWORK', 'TODO_COMPLETED', 'CORE_CIRCLE', 'SUPPORTING_OTHERS',
    'SUFFICIENT_INCOME', 'BMI_RANGE',
]
# Features with valid range [0, 10] — 0 is a valid response (e.g. zero meditation sessions)
LIKERT_0_10 = [
    'TIME_FOR_PASSION', 'ACHIEVEMENT', 'DONATION', 'FLOW',
    'LIVE_VISION', 'PLACES_VISITED', 'FRUITS_VEGGIES',
]
MAX_OOR_RATE = 0.05


def load_data(path='data/Wellbeing_and_lifestyle_data_Kaggle.csv'):
    df = pd.read_csv(path)
    return df


def preprocess(df, use_domain_cleaning=False):
    """
    Target
    ──────
    BURNOUT_RISK = top 30% of (DAILY_STRESS + DAILY_SHOUTING + LOST_VACATION).
    Motivated by Maslach's three burnout dimensions: exhaustion, depersonalisation,
    and reduced accomplishment.  These three columns are excluded from features so
    the model predicts burnout from upstream wellness/lifestyle behaviors.
    This corrects the circular leakage in the original design (WORK_LIFE_BALANCE_SCORE
    had R²=1.0 with all features).

    Intervention 1 – adaptive domain-bounds validation
    ───────────────────────────────────────────────────
    Two Likert-scale families are distinguished:
      [1–10] features  — 0 is impossible (e.g. BMI range, social network)
      [0–10] features  — 0 is a valid response (e.g. no meditation, no hobbies)
    A column is only clamped if its OOR rate is below 5 %, ensuring we fix
    genuine data-entry errors without destroying valid zero responses.

    Intervention 2 – threshold tuning for class imbalance
    ──────────────────────────────────────────────────────
    The ~34 % positive class gives a 1:1.9 imbalance.  Val-set threshold search
    (find_best_threshold in preprocessing_experiment.py) maximises F1 without
    altering the learning objective.
    """
    df = df.drop(columns=['Timestamp'])

    df['GENDER'] = df['GENDER'].map({'Female': 0, 'Male': 1})
    age_map = {'Less than 20': 0, '21 to 35': 1, '36 to 50': 2, '51 or more': 3}
    df['AGE'] = df['AGE'].map(age_map)

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # ── Intervention 1: adaptive domain-bounds validation ─────────────────── #
    if use_domain_cleaning:
        total_fixed, clamped_cols, skipped_cols = 0, [], []

        for col, lo, hi in (
            [(c, 1, 10) for c in LIKERT_1_10] +
            [(c, 0, 10) for c in LIKERT_0_10]
        ):
            if col not in df.columns:
                continue
            oor_mask = (df[col] < lo) | (df[col] > hi)
            oor_rate  = oor_mask.mean()
            if oor_rate == 0:
                continue
            if oor_rate > MAX_OOR_RATE:
                skipped_cols.append(f"{col} ({oor_rate:.1%} OOR — likely different scale)")
                continue
            n = int(oor_mask.sum())
            df[col] = df[col].clip(lo, hi)
            total_fixed += n
            clamped_cols.append(f"{col} ({n} values → [{lo},{hi}])")

        print(f"[domain cleaning] Fixed {total_fixed} errors: "
              f"{', '.join(clamped_cols) or 'none'}")
        if skipped_cols:
            print(f"[domain cleaning] Skipped: {'; '.join(skipped_cols)}")

    # ── Target: burnout symptom composite ────────────────────────────────── #
    burnout_index      = df[BURNOUT_SYMPTOM_COLS].sum(axis=1)
    threshold          = burnout_index.quantile(0.70)
    df['BURNOUT_RISK'] = (burnout_index >= threshold).astype(int)
    pos_rate           = df['BURNOUT_RISK'].mean()
    print(f"[target] threshold={threshold:.1f} | "
          f"positive rate={pos_rate:.1%} | class ratio 1:{(1-pos_rate)/pos_rate:.1f}")

    # ── Feature engineering (no target-adjacent columns used) ─────────────── #
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
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val,   X_test,  y_val,  y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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
    print(f"\nFeatures ({len(cols)}): {cols}")