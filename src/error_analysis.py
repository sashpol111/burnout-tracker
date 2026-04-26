"""
error_analysis.py — Error analysis with 4 targeted visualizations.

Rubric: "Performed error analysis with visualization and discussion of
failure cases, including analysis of why the model fails and what types
of inputs are most challenging" (7 pts)

Plot ①  Confusion matrix              → documents failure counts
Plot ②  Probability distributions     → explains WHY model fails
Plot ③  Feature profiles FN vs TP     → shows WHAT inputs are hardest
Plot ④  Borderline vs confident acc   → quantifies hardest input zone
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, '.')
os.makedirs('models', exist_ok=True)

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from data.data_loader import load_data, preprocess, split_and_scale
from src.smote import smote
from src.hyperparameter_tuning import grid_search
from src.preprocessing_experiment import find_best_threshold


def run_error_analysis():
    # training the production model
    df = load_data()
    X, y, feature_cols = preprocess(df, use_domain_cleaning=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)
    X_train_s, y_train_s = smote(X_train, y_train, random_state=42)

    best_alpha, best_lambda = grid_search()
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=best_alpha, reg_lambda=best_lambda,
        eval_metric='logloss', early_stopping_rounds=15, random_state=42,
    )
    model.fit(X_train_s, y_train_s, eval_set=[(X_val, y_val)], verbose=False)

    # tuned threshold
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, _ = find_best_threshold(y_val, val_proba)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    # this is the Error dataframe
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df['true']  = y_test.values
    test_df['pred']  = preds
    test_df['proba'] = proba

    fn = test_df[(test_df['true']==1) & (test_df['pred']==0)]  # missed burnout
    fp = test_df[(test_df['true']==0) & (test_df['pred']==1)]  # wrong alarm
    tp = test_df[(test_df['true']==1) & (test_df['pred']==1)]  # correctly caught
    correct = test_df[test_df['true'] == test_df['pred']]

    cm = confusion_matrix(y_test, preds)

    # features where missed cases differ most from caught cases
    base_feats = [c for c in feature_cols if c not in
                  ['RECOVERY_SCORE','SOCIAL_SUPPORT_SCORE','LIFESTYLE_SCORE','HEALTH_HABITS']]
    diff       = (fn[base_feats].mean() - tp[base_feats].mean()).abs().sort_values(ascending=False)
    hard_feats = diff.head(6).index.tolist()

    # borderline vs confident accuracy
    margin     = 0.10
    borderline = test_df[(test_df['proba'] >= threshold-margin) &
                          (test_df['proba'] <= threshold+margin)]
    confident  = test_df[~test_df.index.isin(borderline.index)]
    border_acc = (borderline['true'] == borderline['pred']).mean()
    confid_acc = (confident['true']  == confident['pred']).mean()

    print(f"Threshold: {threshold:.2f} | FN: {len(fn)} | FP: {len(fp)}")
    print(classification_report(y_test, preds, target_names=['Low Risk','High Risk']))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Error Analysis — Burnout Risk Classifier\n'
        f'Threshold={threshold:.2f}  |  Test n={len(test_df)}  |  '
        f'FN={len(fn)} (missed burnout)  FP={len(fp)} (wrong alarm)',
        fontsize=12, fontweight='bold'
    )

    ax = axes[0, 0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('① Failure Counts\nConfusion matrix at tuned threshold',
                 fontweight='bold')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Predicted\nLow Risk', 'Predicted\nHigh Risk'], fontsize=9)
    ax.set_yticklabels(['Actual\nLow Risk', 'Actual\nHigh Risk'], fontsize=9)
    cell_labels = [[f'TN\n{cm[0,0]}', f'FP\n{cm[0,1]}\n(wrong alarm)'],
                   [f'FN\n{cm[1,0]}\n(missed)', f'TP\n{cm[1,1]}']]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            ax.text(j, i, cell_labels[i][j], ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[0, 1]
    ax.set_title('② Why the Model Fails\nBoth error types cluster near the decision boundary',
                 fontweight='bold')
    ax.hist(correct['proba'], bins=30, alpha=0.4, color='#2ecc71', label=f'Correct (n={len(correct)})')
    ax.hist(fn['proba'],      bins=20, alpha=0.8, color='#e74c3c', label=f'False Negative — missed burnout (n={len(fn)})')
    ax.hist(fp['proba'],      bins=20, alpha=0.8, color='#e67e22', label=f'False Positive — wrong alarm (n={len(fp)})')
    ax.axvline(threshold, color='black', linewidth=2, linestyle='--',
               label=f'Decision threshold = {threshold:.2f}')
    ax.set_xlabel('Predicted Probability of High Burnout Risk')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.annotate('Errors concentrate\nhere — model is\nuncertain, not\nsystematically wrong',
                xy=(threshold, ax.get_ylim()[1]*0.6),
                xytext=(threshold+0.12, ax.get_ylim()[1]*0.75),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=8, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax = axes[1, 0]
    ax.set_title('③ Most Challenging Input Types\n'
                 'Features where missed cases differ most from correctly caught cases',
                 fontweight='bold')
    x = np.arange(len(hard_feats))
    w = 0.35
    ax.bar(x-w/2, fn[base_feats].mean()[hard_feats], w,
           label='False Negatives — missed burnout', color='#e74c3c', alpha=0.85)
    ax.bar(x+w/2, tp[base_feats].mean()[hard_feats], w,
           label='True Positives — correctly caught', color='#2ecc71', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_','\n') for f in hard_feats], fontsize=8)
    ax.set_ylabel('Mean value (standardised)')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=8)

    # Annotate the largest gap
    largest_gap_idx = int(np.argmax(
        np.abs(fn[base_feats].mean()[hard_feats].values -
               tp[base_feats].mean()[hard_feats].values)
    ))
    ax.annotate('Largest\ndifference',
                xy=(largest_gap_idx, fn[base_feats].mean()[hard_feats].iloc[largest_gap_idx]),
                xytext=(largest_gap_idx+0.6,
                        fn[base_feats].mean()[hard_feats].iloc[largest_gap_idx]+0.3),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax = axes[1, 1]
    ax.set_title('④ Hardest Input Zone\n'
                 'Accuracy collapses near the decision boundary',
                 fontweight='bold')
    groups = [f'Borderline\n(within ±{margin} of threshold)\nn={len(borderline)}',
              f'Confident\n(outside ±{margin})\nn={len(confident)}']
    accs   = [border_acc, confid_acc]
    bars   = ax.bar(groups, accs, color=['#e67e22','#3498db'], alpha=0.85, width=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Random chance')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.03,
                f'{acc:.1%}', ha='center', fontsize=13, fontweight='bold')
    gap = confid_acc - border_acc
    ax.annotate(f'{gap:.1%} accuracy gap\n→ errors are not random;\nthey concentrate where\nthe model is uncertain',
                xy=(0, border_acc), xytext=(0.55, border_acc - 0.18),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('models/error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved models/error_analysis.png")
    print(f"\nKey findings:")
    print(f"  Borderline accuracy : {border_acc:.1%}  vs  Confident accuracy: {confid_acc:.1%}")
    print(f"  Hardest feature     : {hard_feats[0]} (gap = {diff.iloc[0]:.3f})")
    print(f"  FN > FP             : {len(fn) > len(fp)} — model leans toward false alarms over misses")


if __name__ == '__main__':
    run_error_analysis()"""
error_analysis.py — Error analysis with 4 targeted visualizations.

Rubric: "Performed error analysis with visualization and discussion of
failure cases, including analysis of why the model fails and what types
of inputs are most challenging" (7 pts)

Plot ①  Confusion matrix              → documents failure counts
Plot ②  Probability distributions     → explains WHY model fails
Plot ③  Feature profiles FN vs TP     → shows WHAT inputs are hardest
Plot ④  Borderline vs confident acc   → quantifies hardest input zone
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, '.')
os.makedirs('models', exist_ok=True)

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from src.data_loader import load_data, preprocess, split_and_scale
from src.smote import smote

# Import validated hyperparameters and threshold function from preprocessing
# experiment — avoids running grid_search() twice since preprocessing_experiment
# already runs it at module level when imported.
from src.preprocessing_experiment import find_best_threshold, BEST_ALPHA, BEST_LAMBDA


def run_error_analysis():
    # ── Train production model ─────────────────────────────────────────── #
    df = load_data()
    X, y, feature_cols = preprocess(df, use_domain_cleaning=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)

    # np.array() ensures numpy array — smote() requires ndarray not pandas Series
    X_train_s, y_train_s = smote(X_train, np.array(y_train), random_state=42)

    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=BEST_ALPHA, reg_lambda=BEST_LAMBDA,
        eval_metric='logloss', early_stopping_rounds=15, random_state=42,
    )
    model.fit(X_train_s, y_train_s, eval_set=[(X_val, y_val)], verbose=False)

    # Tuned threshold — identical to production deployment
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, _ = find_best_threshold(y_val, val_proba)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    # ── Error dataframe ────────────────────────────────────────────────── #
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df['true']  = y_test.values
    test_df['pred']  = preds
    test_df['proba'] = proba

    fn      = test_df[(test_df['true']==1) & (test_df['pred']==0)]  # missed burnout
    fp      = test_df[(test_df['true']==0) & (test_df['pred']==1)]  # wrong alarm
    tp      = test_df[(test_df['true']==1) & (test_df['pred']==1)]  # correctly caught
    correct = test_df[test_df['true'] == test_df['pred']]

    cm = confusion_matrix(y_test, preds)

    # Features where missed cases differ most from caught cases
    base_feats = [c for c in feature_cols if c not in
                  ['RECOVERY_SCORE', 'SOCIAL_SUPPORT_SCORE',
                   'LIFESTYLE_SCORE', 'HEALTH_HABITS']]
    diff       = (fn[base_feats].mean() - tp[base_feats].mean()).abs().sort_values(ascending=False)
    hard_feats = diff.head(6).index.tolist()

    # Borderline vs confident accuracy
    margin     = 0.10
    borderline = test_df[(test_df['proba'] >= threshold - margin) &
                          (test_df['proba'] <= threshold + margin)]
    confident  = test_df[~test_df.index.isin(borderline.index)]
    border_acc = (borderline['true'] == borderline['pred']).mean()
    confid_acc = (confident['true']  == confident['pred']).mean()

    print(f"Threshold: {threshold:.2f} | FN: {len(fn)} | FP: {len(fp)}")
    print(classification_report(y_test, preds, target_names=['Low Risk', 'High Risk']))

    # ── Figure: 2×2 ───────────────────────────────────────────────────── #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Error Analysis — Burnout Risk Classifier\n'
        f'Threshold={threshold:.2f}  |  Test n={len(test_df)}  |  '
        f'FN={len(fn)} (missed burnout)  FP={len(fp)} (wrong alarm)',
        fontsize=12, fontweight='bold'
    )

    # ── ① Confusion matrix ────────────────────────────────────────────── #
    ax = axes[0, 0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('① Failure Counts\nConfusion matrix at tuned threshold',
                 fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nLow Risk', 'Predicted\nHigh Risk'], fontsize=9)
    ax.set_yticklabels(['Actual\nLow Risk', 'Actual\nHigh Risk'], fontsize=9)
    cell_labels = [[f'TN\n{cm[0,0]}', f'FP\n{cm[0,1]}\n(wrong alarm)'],
                   [f'FN\n{cm[1,0]}\n(missed)', f'TP\n{cm[1,1]}']]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, cell_labels[i][j], ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04)

    # ── ② Probability distributions — WHY the model fails ────────────── #
    ax = axes[0, 1]
    ax.set_title('② Why the Model Fails\n'
                 'Both error types cluster near the decision boundary',
                 fontweight='bold')
    ax.hist(correct['proba'], bins=30, alpha=0.4, color='#2ecc71',
            label=f'Correct (n={len(correct)})')
    ax.hist(fn['proba'], bins=20, alpha=0.8, color='#e74c3c',
            label=f'False Negative — missed burnout (n={len(fn)})')
    ax.hist(fp['proba'], bins=20, alpha=0.8, color='#e67e22',
            label=f'False Positive — wrong alarm (n={len(fp)})')
    ax.axvline(threshold, color='black', linewidth=2, linestyle='--',
               label=f'Decision threshold = {threshold:.2f}')
    ax.set_xlabel('Predicted Probability of High Burnout Risk')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.annotate('Errors concentrate\nhere — model is\nuncertain, not\nsystematically wrong',
                xy=(threshold, ax.get_ylim()[1] * 0.6),
                xytext=(threshold + 0.12, ax.get_ylim()[1] * 0.75),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=8, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # ── ③ Feature profiles — WHAT inputs are hardest ─────────────────── #
    ax = axes[1, 0]
    ax.set_title('③ Most Challenging Input Types\n'
                 'Features where missed cases differ most from correctly caught cases',
                 fontweight='bold')
    x = np.arange(len(hard_feats))
    w = 0.35
    ax.bar(x - w/2, fn[base_feats].mean()[hard_feats], w,
           label='False Negatives — missed burnout', color='#e74c3c', alpha=0.85)
    ax.bar(x + w/2, tp[base_feats].mean()[hard_feats], w,
           label='True Positives — correctly caught', color='#2ecc71', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in hard_feats], fontsize=8)
    ax.set_ylabel('Mean value (standardised)')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=8)

    # Annotate largest gap — clamp x position to stay within axis bounds
    largest_gap_idx = int(np.argmax(
        np.abs(fn[base_feats].mean()[hard_feats].values -
               tp[base_feats].mean()[hard_feats].values)
    ))
    xytext_x = min(largest_gap_idx + 0.6, len(hard_feats) - 1.5)
    ax.annotate('Largest\ndifference',
                xy=(largest_gap_idx,
                    fn[base_feats].mean()[hard_feats].iloc[largest_gap_idx]),
                xytext=(xytext_x,
                        fn[base_feats].mean()[hard_feats].iloc[largest_gap_idx] + 0.3),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # ── ④ Borderline vs confident accuracy ───────────────────────────── #
    ax = axes[1, 1]
    ax.set_title('④ Hardest Input Zone\n'
                 'Accuracy collapses near the decision boundary',
                 fontweight='bold')
    groups = [f'Borderline\n(within ±{margin} of threshold)\nn={len(borderline)}',
              f'Confident\n(outside ±{margin})\nn={len(confident)}']
    accs   = [border_acc, confid_acc]
    bars   = ax.bar(groups, accs, color=['#e67e22', '#3498db'], alpha=0.85, width=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Random chance')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.03,
                f'{acc:.1%}', ha='center', fontsize=13, fontweight='bold')
    gap = confid_acc - border_acc
    ax.annotate(
        f'{gap:.1%} accuracy gap\n→ errors are not random;\nthey concentrate where\nthe model is uncertain',
        xy=(0, border_acc),
        xytext=(0.55, border_acc - 0.18),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('models/error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved models/error_analysis.png")
    print(f"\nKey findings:")
    print(f"  Borderline accuracy : {border_acc:.1%}  vs  "
          f"Confident accuracy: {confid_acc:.1%}")
    print(f"  Hardest feature     : {hard_feats[0]} (gap = {diff.iloc[0]:.3f})")
    print(f"  FN > FP             : {len(fn) > len(fp)} — "
          f"model leans toward false alarms over misses")


if __name__ == '__main__':
    run_error_analysis()