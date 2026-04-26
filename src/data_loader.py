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
# Features with valid range [0, 10] — 0 is a valid response
LIKERT_0_10 = [
    'TIME_FOR_PASSION', 'ACHIEVEMENT', 'DONATION', 'FLOW',
    'LIVE_VISION', 'PLACES_VISITED', 'FRUITS_VEGGIES',
]
MAX_OOR_RATE = 0.05


def load_data(path='data/unified_dataset.csv',
              use_unified=False, force_rebuild=False):
    """
    Load training data. Two modes:

    use_unified=False (default):
        Load the Kaggle wellness survey CSV directly.
        Fast, no API calls needed. preprocess() will encode categoricals,
        derive the burnout label, and engineer features.

    use_unified=True:
        Build the unified dataset from all three API sources:
          - Kaggle API    (structured wellness survey)
          - Groq API      (LLM-generated synthetic profiles)
          - HuggingFace   (annotated mental health posts)
        The unified dataset is pre-processed (numeric features, BURNOUT_RISK
        already computed). preprocess() detects this and skips those steps.
        Results cached to data/unified_dataset.csv after first build.
        Pass force_rebuild=True to re-fetch from APIs.
    """
    if use_unified:
        from src.data_pipeline import build_unified_dataset
        df = build_unified_dataset(force_rebuild=force_rebuild)
        return df.drop(columns=['source'], errors='ignore')
    df = pd.read_csv(path)
    return df


def preprocess(df, use_domain_cleaning=False):
    """
    Preprocessing pipeline — handles both Kaggle raw data and unified dataset.

    Detects which source it's working with:
      - Kaggle raw: has 'Timestamp', string GENDER/AGE, symptom columns
      - Unified:    already numeric, BURNOUT_RISK already computed

    Intervention 1 – adaptive domain-bounds validation
    ───────────────────────────────────────────────────
    Likert-scale columns clamped to valid ranges [0,10] or [1,10].
    Adaptive guard skips columns where >5% OOR (different natural scale).

    Intervention 2 – threshold tuning for class imbalance
    ──────────────────────────────────────────────────────
    Applied in preprocessing_experiment.py on the val set only.
    """

    # ── Detect source and normalise accordingly ───────────────────────── #
    # Kaggle raw: has Timestamp column, string GENDER/AGE, symptom columns
    # Unified:    already numeric, BURNOUT_RISK already present, no Timestamp
    is_kaggle_raw = (
        'Timestamp' in df.columns or
        ('GENDER' in df.columns and df['GENDER'].dtype == object) or
        ('BURNOUT_RISK' not in df.columns and
         all(c in df.columns for c in BURNOUT_SYMPTOM_COLS))
    )

    if is_kaggle_raw:
        # ── Kaggle raw data: needs full encoding + label derivation ──── #
        df = df.drop(columns=['Timestamp'], errors='ignore')

        if 'GENDER' in df.columns and df['GENDER'].dtype == object:
            df['GENDER'] = df['GENDER'].map({'Female': 0, 'Male': 1})

        if 'AGE' in df.columns and df['AGE'].dtype == object:
            age_map = {'Less than 20': 0, '21 to 35': 1,
                       '36 to 50': 2,    '51 or more': 3}
            df['AGE'] = df['AGE'].map(age_map)

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        # Domain cleaning on raw survey data
        if use_domain_cleaning:
            _apply_domain_cleaning(df)

        # Derive burnout label from symptom composite
        burnout_index      = df[BURNOUT_SYMPTOM_COLS].sum(axis=1)
        threshold          = burnout_index.quantile(0.70)
        df['BURNOUT_RISK'] = (burnout_index >= threshold).astype(int)
        pos_rate           = df['BURNOUT_RISK'].mean()
        print(f"[target] threshold={threshold:.1f} | "
              f"positive rate={pos_rate:.1%} | "
              f"class ratio 1:{(1-pos_rate)/pos_rate:.1f}")

    else:
        # ── Unified dataset: already numeric, BURNOUT_RISK already set ── #
        df = df.drop(columns=['source'], errors='ignore')  # text col → drop before coerce
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        # Domain cleaning still valid — synthetic/HF values may drift
        if use_domain_cleaning:
            _apply_domain_cleaning(df)

        pos_rate = df['BURNOUT_RISK'].mean()
        print(f"[unified dataset] {len(df)} rows | "
              f"positive rate={pos_rate:.1%} | "
              f"class ratio 1:{(1-pos_rate)/pos_rate:.1f}")

    # ── Feature engineering (same for both sources) ───────────────────── #
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


def _apply_domain_cleaning(df):
    """
    Adaptive domain-bounds validation — modifies df in place.
    Only clamps columns where OOR rate < MAX_OOR_RATE (genuine errors,
    not a different natural scale).
    """
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
    print("=== Testing with Kaggle source ===")
    df = load_data()
    X, y, cols = preprocess(df, use_domain_cleaning=True)
    split_and_scale(X, y)
    print(f"Features ({len(cols)}): {cols}")

    print("\n=== Testing with unified source ===")
    if __import__('os').path.exists('data/unified_dataset.csv'):
        df2 = load_data(use_unified=True)
        X2, y2, cols2 = preprocess(df2, use_domain_cleaning=True)
        split_and_scale(X2, y2)
    else:
        print("unified_dataset.csv not found — run src/data_pipeline.py first")