# Grid search over XGBoost L1/L2 regularization using the validation set.

import numpy as np
import itertools
import sys
sys.path.insert(0, '.')

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from data.data_loader import load_data, preprocess, split_and_scale
from src.smote import smote


def grid_search():
    df = load_data()
    X, y, _ = preprocess(df.copy(), use_domain_cleaning=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)

    # SMOTE only on training set — val and test stay as-is
    X_train_s, y_train_s = smote(X_train, np.array(y_train), random_state=42)

    # 3 values each = 9 combinations
    alphas  = [0.0, 0.1, 1.0]   # L1
    lambdas = [0.1, 1.0, 5.0]   # L2

    BASE_PARAMS = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        eval_metric='logloss', early_stopping_rounds=15,
        random_state=42,
    )

    print(f"  {'reg_alpha':>10} | {'reg_lambda':>10} | {'val AUC':>8} | {'trees':>6}")
    print("  " + "-" * 44)

    results = []
    for alpha, lam in itertools.product(alphas, lambdas):
        model = XGBClassifier(**BASE_PARAMS, reg_alpha=alpha, reg_lambda=lam)
        model.fit(X_train_s, y_train_s, eval_set=[(X_val, y_val)], verbose=False)

        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        trees   = model.best_iteration + 1
        results.append((alpha, lam, val_auc, trees, model))
        print(f"  {alpha:>10.1f} | {lam:>10.1f} | {val_auc:>8.4f} | {trees:>6}")

    best_alpha, best_lam, best_val_auc, best_trees, best_model = max(results, key=lambda x: x[2])
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    print(f"\n  best: reg_alpha={best_alpha}, reg_lambda={best_lam} | "
          f"val AUC={best_val_auc:.4f} | test AUC={test_auc:.4f}")

    return best_alpha, best_lam


if __name__ == '__main__':
    grid_search()