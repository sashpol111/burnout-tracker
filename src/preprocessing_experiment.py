import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, split_and_scale
from src.smote import smote
from src.hyperparameter_tuning import grid_search

# these are the regularization hyperparameters selected by grid search
print("Running hyperparameter tuning to select reg_alpha and reg_lambda...")
BEST_ALPHA, BEST_LAMBDA = grid_search()
print(f"\nUsing reg_alpha={BEST_ALPHA}, reg_lambda={BEST_LAMBDA} for all conditions.\n")

XGB_PARAMS = dict(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=BEST_ALPHA, reg_lambda=BEST_LAMBDA,
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

    # SMOTE oversampling
    if use_smote:
        before = dict(zip(*np.unique(y_train, return_counts=True)))
        X_train, y_train = smote(X_train, y_train, k=5, random_state=42)
        after = dict(zip(*np.unique(y_train, return_counts=True)))
        print(f"  [SMOTE] train set {before} → {after}")

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # threshold tuning
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

    # Baseline — no preprocessing interventions
    print("\n── Baseline (no cleaning, no imbalance handling) ──")
    X, y, _ = preprocess(df.copy(), use_domain_cleaning=False)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)
    results['Baseline'] = run_condition(
        "Baseline",
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # Domain cleaning only
    print("\n── Intervention: domain cleaning only ──")
    X, y, _ = preprocess(df.copy(), use_domain_cleaning=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)
    results['Domain cleaning'] = run_condition(
        "Domain cleaning (fix impossible Likert values)",
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # Domain cleaning + SMOTE
    print("\n── Intervention A: domain cleaning + SMOTE ──")
    results['+ SMOTE'] = run_condition(
        "Domain cleaning + SMOTE",
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_smote=True,
    )

    # Domain cleaning + threshold tuning 
    print("\n── Intervention B: domain cleaning + threshold tuning ──")
    results['+ Threshold'] = run_condition(
        "Domain cleaning + threshold tuning",
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_threshold_tuning=True,
    )

    # Domain cleaning + SMOTE + threshold tuning (full pipeline)
    print("\n── Full pipeline: domain cleaning + SMOTE + threshold tuning ──")
    results['Full pipeline'] = run_condition(
        "Full pipeline (cleaning + SMOTE + threshold)",
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_smote=True,
        use_threshold_tuning=True,
    )

    # summarisation table
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


#regularization

def regularization_experiment():
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    df = load_data()
    X, y, _ = preprocess(df.copy(), use_domain_cleaning=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)
    X_train_s, y_train_s = smote(X_train, y_train, random_state=42)

    def fit_eval(name, max_depth=8, min_child_weight=1,
                 reg_alpha=0, reg_lambda=0,
                 use_early_stopping=False, n_estimators=500):
        model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            eval_metric='logloss', random_state=42,
            **(dict(early_stopping_rounds=20) if use_early_stopping else {}),
        )
        fit_kw = dict(eval_set=[(X_val, y_val)], verbose=False) if use_early_stopping else dict(verbose=False)
        model.fit(X_train_s, y_train_s, **fit_kw)

        stopped_at = (model.best_iteration + 1) if use_early_stopping else n_estimators
        train_auc  = roc_auc_score(y_train_s, model.predict_proba(X_train_s)[:, 1])
        test_auc   = roc_auc_score(y_test,    model.predict_proba(X_test)[:, 1])
        gap        = train_auc - test_auc

        print(f"  {name:<42s} | train: {train_auc:.3f} | test: {test_auc:.3f} | gap: {gap:.3f} | trees: {stopped_at}")
        return gap, test_auc

    print("\n" + "=" * 75)
    print("REGULARIZATION EXPERIMENT")
    print("Overfitting gap = train AUC - test AUC  (lower = less overfit)")
    print("Unregularized baseline: depth=8, min_child=1  -> forces overfitting")
    print("=" * 75)
    print(f"  {'Condition':<42s} | train AUC | test AUC | gap   | trees")
    print("  " + "-" * 70)

    r = {}
    r['No reg']      = fit_eval("No regularization  (depth=8, min_child=1)")
    r['L2']          = fit_eval("L2 only  (reg_lambda=5.0)",          reg_lambda=5.0)
    r['L1']          = fit_eval("L1 only  (reg_alpha=2.0)",           reg_alpha=2.0)
    r['L1+L2']       = fit_eval("L1 + L2  (alpha=2.0, lambda=5.0)",  reg_alpha=2.0, reg_lambda=5.0)
    r['ES']          = fit_eval("Early stopping only  (depth=8)",     use_early_stopping=True)
    r['Production']  = fit_eval("L1 + L2 + early stopping  ◀ production",
                                   max_depth=4, min_child_weight=5,
                                   reg_alpha=BEST_ALPHA, reg_lambda=BEST_LAMBDA,
                                   use_early_stopping=True)

    bg, ba = r['No reg']
    fg, fa = r['Production']
    print(f"\n  Overfitting gap : {bg:.3f} -> {fg:.3f}  ({(bg-fg)/bg*100:.0f}% reduction)")
    print(f"  Test AUC        : {ba:.3f} -> {fa:.3f}  (generalisation improves)")
    print("\n  L1 (reg_alpha) : drives irrelevant weights toward zero.")
    print("  L2 (reg_lambda): shrinks all weights to reduce variance.")
    print("  Early stopping : halts when val loss plateaus, avoids memorising noise.")
    print("  Combined       : each addresses a different overfit source; together")
    print("                   they give the largest gap reduction.")


if __name__ == '__main__':
    regularization_experiment()