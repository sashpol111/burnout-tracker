import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, split_and_scale
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == '__main__':
    df = load_data()
    X, y, feature_cols = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)
    
    model = joblib.load('models/xgboost_model.pkl')
    
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives (correctly said no burnout): {cm[0][0]}")
    print(f"False Positives (wrongly flagged burnout): {cm[0][1]}")
    print(f"False Negatives (missed burnout cases): {cm[1][0]}")
    print(f"True Positives (correctly caught burnout): {cm[1][1]}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['No Burnout', 'Burnout']))
    
    # Analyze false negatives - missed burnout cases
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    X_test_df['true'] = y_test.values
    X_test_df['pred'] = preds
    X_test_df['proba'] = proba
    
    false_negatives = X_test_df[(X_test_df['true'] == 1) & (X_test_df['pred'] == 0)]
    false_positives = X_test_df[(X_test_df['true'] == 0) & (X_test_df['pred'] == 1)]
    
    print(f"\nFalse Negatives (missed burnout): {len(false_negatives)}")
    print(f"False Positives (wrong burnout flag): {len(false_positives)}")
    
    print("\nAvg profile of MISSED burnout cases (false negatives):")
    print(false_negatives[feature_cols].mean().sort_values().head(8))
    
    print("\nAvg profile of WRONGLY FLAGGED cases (false positives):")
    print(false_positives[feature_cols].mean().sort_values(ascending=False).head(8))
    
    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['No Burnout', 'Burnout'])
    axes[0].set_yticklabels(['No Burnout', 'Burnout'])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i][j]), ha='center', va='center', 
                        color='white' if cm[i][j] > cm.max()/2 else 'black', fontsize=14)
    
    # Plot error distribution
    fn_proba = false_negatives['proba'].values
    fp_proba = false_positives['proba'].values
    axes[1].hist(fn_proba, bins=20, alpha=0.7, label=f'False Negatives (n={len(fn_proba)})', color='red')
    axes[1].hist(fp_proba, bins=20, alpha=0.7, label=f'False Positives (n={len(fp_proba)})', color='blue')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('models/error_analysis.png', dpi=150)
    plt.close()
    print("\nSaved error_analysis.png")