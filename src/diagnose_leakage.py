# Documents the design decision to replace WORK_LIFE_BALANCE_SCORE with a burnout composite target.
# Run with: python src/diagnose_leakage.py
# Run with force redownload: python src/diagnose_leakage.py --redownload

import os
import sys
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')

BURNOUT_SYMPTOM_COLS = ['DAILY_STRESS', 'DAILY_SHOUTING', 'LOST_VACATION']
KAGGLE_PATH = 'data/Wellbeing_and_lifestyle_data_Kaggle.csv'


def download_kaggle_csv():

    username = os.getenv('KAGGLE_USERNAME')
    key      = os.getenv('KAGGLE_KEY')
    if not username or not key:
        raise RuntimeError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in your .env file.\n"
            "Get them from kaggle.com/settings/account -> Create New API Token."
        )

    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY']      = key

    import kagglehub
    print("Downloading from Kaggle API...")
    dl   = kagglehub.dataset_download('ydalat/lifestyle-and-wellbeing-data')
    csvs = glob.glob(os.path.join(dl, '**', '*.csv'), recursive=True)
    if not csvs:
        raise FileNotFoundError("No CSV found in downloaded dataset")
    os.makedirs('data', exist_ok=True)
    shutil.copy(csvs[0], KAGGLE_PATH)
    print(f"  Saved to {KAGGLE_PATH}")


def load_raw_kaggle(force=False):
    if force or not os.path.exists(KAGGLE_PATH):
        download_kaggle_csv()
    else:
        print(f"Using cached file: {KAGGLE_PATH}")
    return pd.read_csv(KAGGLE_PATH)


def run_leakage_diagnostic(force_download=False):
    df = load_raw_kaggle(force=force_download)

    df = df.drop(columns=['Timestamp'], errors='ignore')
    df['GENDER'] = df['GENDER'].map({'Female': 0, 'Male': 1})
    df['AGE']    = df['AGE'].map({'Less than 20': 0, '21 to 35': 1,
                                   '36 to 50': 2,   '51 or more': 3})
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    all_features = [c for c in df.columns
                    if c not in ['WORK_LIFE_BALANCE_SCORE'] + BURNOUT_SYMPTOM_COLS]

    # Test 1: can features linearly reconstruct WORK_LIFE_BALANCE_SCORE?
    X    = df[all_features].values
    y    = df['WORK_LIFE_BALANCE_SCORE'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    r2   = r2_score(y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te))

    print("=" * 60)
    print("LEAKAGE DIAGNOSTIC")
    print("=" * 60)
    print(f"\nTest 1 — Linear R² of WORK_LIFE_BALANCE_SCORE ~ features:")
    print(f"  R² = {r2:.4f}")
    print(f"  {'LEAK CONFIRMED — trivial classification task' if r2 > 0.99 else 'No leakage detected'}")

    # Test 2: can features reconstruct the burnout composite?
    burnout_index = df[BURNOUT_SYMPTOM_COLS].sum(axis=1).values
    feat_vals     = df[all_features].values
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        feat_vals, burnout_index, test_size=0.2, random_state=42)
    r2_new = r2_score(y_te2, LinearRegression().fit(X_tr2, y_tr2).predict(X_te2))

    print(f"\nTest 2 — Linear R² of burnout_index ~ wellness features:")
    print(f"  R² = {r2_new:.4f}")
    print(f"  {'OK — genuine prediction task' if r2_new < 0.5 else 'WARNING — check target construction'}")

    # Test 3: class balance of new binary target (top 30%)
    threshold    = pd.Series(burnout_index).quantile(0.70)
    burnout_risk = (burnout_index >= threshold).astype(int)
    pos_rate     = burnout_risk.mean()

    print(f"\nTest 3 — New target class distribution:")
    print(f"  Threshold (70th pct): {threshold:.1f}")
    print(f"  Positive rate       : {pos_rate:.1%}")
    print(f"  Class ratio         : 1:{(1-pos_rate)/pos_rate:.1f}")

    print(f"\n{'='*60}")
    print("DESIGN DECISION SUMMARY")
    print(f"{'='*60}")
    print(f"  Original target : WORK_LIFE_BALANCE_SCORE  (R²={r2:.4f})")
    print(f"  Problem         : exact linear leakage — trivial to predict")
    print(f"  New target      : top-30% of burnout symptom composite")
    print(f"                    (DAILY_STRESS + DAILY_SHOUTING + LOST_VACATION)")
    print(f"  New R²          : {r2_new:.4f} — genuine predictive challenge")
    print(f"  Justification   : Maslach Burnout Inventory dimensions")
    print(f"                    (exhaustion, depersonalisation, reduced accomplishment)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--redownload', action='store_true',
                        help='Force re-download of Kaggle CSV even if cached')
    args = parser.parse_args()
    run_leakage_diagnostic(force_download=args.redownload)