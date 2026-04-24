import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, split_and_scale

def plot_feature_importance(model, feature_cols):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_cols)), importance[indices])
    plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance - XGBoost')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150)
    plt.close()
    print("Saved feature_importance.png")
    
    print("\nTop 5 burnout predictors:")
    for i in range(5):
        print(f"  {feature_cols[indices[i]]}: {importance[indices[i]]:.3f}")

def plot_training_curves(train_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neural Network Training Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150)
    plt.close()
    print("Saved training_curves.png")

def plot_model_comparison():
    models = ['Baseline', 'XGBoost', 'Neural Net']
    accuracy = [0.755, 0.951, 0.966]
    f1 = [0.000, 0.893, 0.935]
    auc = [0.500, 0.991, 0.999]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, accuracy, width, label='Accuracy')
    plt.bar(x, f1, width, label='F1')
    plt.bar(x + width, auc, width, label='AUC')
    plt.xticks(x, models)
    plt.ylim(0, 1.1)
    plt.title('Model Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=150)
    plt.close()
    print("Saved model_comparison.png")

if __name__ == '__main__':
    xgb_model = joblib.load('models/xgboost_model.pkl')
    train_losses = joblib.load('models/train_losses.pkl')
    _, _, feature_cols = preprocess(load_data())
    
    plot_feature_importance(xgb_model, feature_cols)
    plot_training_curves(train_losses)
    plot_model_comparison()