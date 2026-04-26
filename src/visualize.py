"""
visualize.py

Produces standalone visualizations from the live trained model.
Saves all plots to models/ for inclusion in README and documentation.

Plots:
  1. Feature importances (from trained XGBoost)
  2. Model comparison bar chart (from preprocessing_experiment results)
  3. Risk score distribution (from test set predictions)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, '.')
from data.data_loader import load_data, preprocess, split_and_scale

def plot_feature_importance(model, feature_cols):
    """Bar chart of XGBoost feature importances — live from trained model."""
    importance = model.feature_importances_
    indices    = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(feature_cols)),
           importance[indices],
           color='steelblue', alpha=0.85)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in indices],
                        rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — XGBoost Production Model',
                 fontweight='bold')
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150)
    plt.close()
    print("Saved models/feature_importance.png")

    print("\nTop 5 burnout predictors:")
    for i in range(5):
        print(f"  {feature_cols[indices[i]]}: {importance[indices[i]]:.3f}")


def plot_model_comparison(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Live model comparison — metrics computed from actual model runs,
    not hardcoded. Compares baseline (no interventions) vs full pipeline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    results = {}

    # Baseline — XGBoost, threshold=0.5, no SMOTE
    model_base = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=BEST_ALPHA, reg_lambda=BEST_LAMBDA,
        eval_metric='logloss', early_stopping_rounds=15, random_state=42,
    )
    model_base.fit(X_train, np.array(y_train),
                   eval_set=[(X_val, y_val)], verbose=False)
    base_proba = model_base.predict_proba(X_test)[:, 1]
    base_preds = (base_proba >= 0.5).astype(int)
    results['Baseline\n(no interventions)'] = {
        'F1':  f1_score(y_test, base_preds, zero_division=0),
        'AUC': roc_auc_score(y_test, base_proba),
        'Acc': accuracy_score(y_test, base_preds),
    }

    # Full pipeline — SMOTE + tuned threshold
    model_full = train_model(X_train, y_train, X_val, y_val)
    full_proba  = model_full.predict_proba(X_test)[:, 1]
    val_proba   = model_full.predict_proba(X_val)[:, 1]
    threshold, _ = find_best_threshold(y_val, val_proba)
    full_preds  = (full_proba >= threshold).astype(int)
    results['Full Pipeline\n(SMOTE + threshold)'] = {
        'F1':  f1_score(y_test, full_preds, zero_division=0),
        'AUC': roc_auc_score(y_test, full_proba),
        'Acc': accuracy_score(y_test, full_preds),
    }

    # Logistic Regression — full pipeline
    from sklearn.linear_model import LogisticRegression
    X_train_s, y_train_s = smote(X_train, np.array(y_train), random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train_s)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_thresh, _ = find_best_threshold(y_val, lr.predict_proba(X_val)[:, 1])
    lr_preds = (lr_proba >= lr_thresh).astype(int)
    results['Logistic\nRegression'] = {
        'F1':  f1_score(y_test, lr_preds, zero_division=0),
        'AUC': roc_auc_score(y_test, lr_proba),
        'Acc': accuracy_score(y_test, lr_preds),
    }

    # Plot
    labels  = list(results.keys())
    metrics = ['F1', 'AUC', 'Acc']
    colors  = ['#e74c3c', '#3498db', '#2ecc71']
    x       = np.arange(len(labels))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[l][metric] for l in labels]
        bars = ax.bar(x + i*width - width, vals, width,
                      label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.2f}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Live Results\n'
                 '(all metrics computed from held-out test set)',
                 fontweight='bold')
    ax.legend()
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8,
               label='Random baseline')
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=150)
    plt.close()
    print("Saved models/model_comparison.png")

    print("\nModel comparison results:")
    for label, metrics_dict in results.items():
        label_clean = label.replace('\n', ' ')
        print(f"  {label_clean:35s} | "
              f"F1: {metrics_dict['F1']:.3f} | "
              f"AUC: {metrics_dict['AUC']:.3f} | "
              f"Acc: {metrics_dict['Acc']:.3f}")


def plot_risk_distribution(model, X_test, y_test, threshold):
    """Risk score distribution on test set — shows model calibration."""
    proba = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(proba[y_test == 0], bins=30, alpha=0.6,
            color='#2ecc71', label='True Low Risk')
    ax.hist(proba[y_test == 1], bins=30, alpha=0.6,
            color='#e74c3c', label='True High Risk')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Decision threshold = {threshold:.2f}')
    ax.set_xlabel('Predicted Burnout Risk Score')
    ax.set_ylabel('Count')
    ax.set_title('Risk Score Distribution by True Label\n'
                 'Good separation = model distinguishes risk levels',
                 fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('models/risk_distribution.png', dpi=150)
    plt.close()
    print("Saved models/risk_distribution.png")


if __name__ == '__main__':
    df = load_data()
    X, y, feature_cols = preprocess(df, use_domain_cleaning=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)

    print("Training production model for visualization...")
    model = train_model(X_train, y_train, X_val, y_val)

    val_proba        = model.predict_proba(X_val)[:, 1]
    threshold, _     = find_best_threshold(y_val, val_proba)

    plot_feature_importance(model, feature_cols)
    plot_model_comparison(X_train, y_train, X_val, y_val, X_test, y_test)
    plot_risk_distribution(model, X_test, y_test, threshold)

    print("\nAll visualizations saved to models/")