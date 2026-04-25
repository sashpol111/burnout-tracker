import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, split_and_scale
from src.smote import smote

# ── Shared hyperparameters — identical across all conditions ──────────────── #
# No scale_pos_weight: imbalance is handled explicitly by SMOTE and/or
# threshold tuning so the model itself stays unmodified between conditions.
XGB_PARAMS = dict(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='logloss', early_stopping_rounds=15,
    random_state=42,
)


def find_best_threshold(y_true, proba):
    """
    Intervention B — inference-time class imbalance correction.
    Search val-set for the threshold that maximises F1.
    Test set is never touched during this search.
    """
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (proba >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def evaluate(model, X, y, name, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, zero_division=0)
    auc   = roc_auc_score(y, proba)
    suffix = f"  (thresh={threshold:.2f})" if threshold != 0.5 else ""
    print(f"  {name:48s} | F1: {f1:.3f} | AUC: {auc:.3f} | Acc: {acc:.3f}{suffix}")
    return acc, f1, auc


def run_condition(label, X_train, y_train, X_val, y_val, X_test, y_test,
                  use_smote=False, use_threshold_tuning=False):
    """Train one condition and evaluate it; return (f1, auc) for summary table."""

    # ── Intervention A: SMOTE oversampling ───────────────────────────────── #
    if use_smote:
        before = dict(zip(*np.unique(y_train, return_counts=True)))
        X_train, y_train = smote(X_train, y_train, k=5, random_state=42)
        after = dict(zip(*np.unique(y_train, return_counts=True)))
        print(f"  [SMOTE] train set {before} → {after}")

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ── Intervention B: threshold tuning ─────────────────────────────────── #
    threshold = 0.5
    if use_threshold_tuning:
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = find_best_threshold(y_val, val_proba)
        print(f"  [threshold search] best val F1={val_f1:.3f} at thresh={threshold:.2f}")

    _, f1, auc = evaluate(model, X_test, y_test, label, threshold=threshold)
    return f1, auc


if __name__ == '__main__':
    df = load_data()

    print("\n" + "=" * 72)
    print("PREPROCESSING EXPERIMENT")
    print("Target: top-30% of (DAILY_STRESS + DAILY_SHOUTING + LOST_VACATION)")
    print("Two interventions: A=SMOTE oversampling  B=threshold tuning")
    print("=" * 72)

    results = {}

    # ── 1. Baseline — no preprocessing interventions ─────────────────────── #
    print("\n── Baseline (no cleaning, no imbalance handling) ──")
    X, y, _ = preprocess(df.copy(), use_domain_cleaning=False)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)
    results['Baseline'] = run_condition(
        "Baseline",
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # ── 2. Domain cleaning only ───────────────────────────────────────────── #
    print("\n── Intervention: domain cleaning only ──")
    X, y, _ = preprocess(df.copy(), use_domain_cleaning=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)
    results['Domain cleaning'] = run_condition(
        "Domain cleaning (fix impossible Likert values)",
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # ── 3. Domain cleaning + SMOTE ────────────────────────────────────────── #
    print("\n── Intervention A: domain cleaning + SMOTE ──")
    results['+ SMOTE'] = run_condition(
        "Domain cleaning + SMOTE",
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_smote=True,
    )

    # ── 4. Domain cleaning + threshold tuning ────────────────────────────── #
    print("\n── Intervention B: domain cleaning + threshold tuning ──")
    results['+ Threshold'] = run_condition(
        "Domain cleaning + threshold tuning",
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_threshold_tuning=True,
    )

    # ── 5. Domain cleaning + SMOTE + threshold tuning (full pipeline) ─────── #
    print("\n── Full pipeline: domain cleaning + SMOTE + threshold tuning ──")
    results['Full pipeline'] = run_condition(
        "Full pipeline (cleaning + SMOTE + threshold)",
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_smote=True,
        use_threshold_tuning=True,
    )

    # ── Summary table ─────────────────────────────────────────────────────── #
    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  {'Condition':<40} | {'F1':>6} | {'AUC':>6}")
    print("  " + "-" * 58)
    for name, (f1, auc) in results.items():
        marker = " ◀ best F1" if f1 == max(v[0] for v in results.values()) else ""
        print(f"  {name:<40} | {f1:>6.3f} | {auc:>6.3f}{marker}")

    baseline_f1 = results['Baseline'][0]
    best_f1     = max(v[0] for v in results.values())
    print(f"\n  F1 improvement baseline → full pipeline: "
          f"{baseline_f1:.3f} → {best_f1:.3f} "
          f"(+{best_f1 - baseline_f1:.3f}, "
          f"{(best_f1 - baseline_f1) / baseline_f1 * 100:.0f}% relative)")

    print("\n── What each intervention addresses ──")
    print("  Domain cleaning : fixes impossible Likert values (data quality)")
    print("  SMOTE           : synthesises minority-class examples (training-time imbalance)")
    print("  Threshold tuning: shifts decision boundary on val set (inference-time imbalance)")
    print("  SMOTE + threshold: complementary — one fixes training distribution,")
    print("                    the other optimises the operating point.")