import numpy as np
from xgboost import XGBClassifier
from data.data_loader import load_data, preprocess, split_and_scale
from src.smote import smote


def train_model():
    df = load_data()
    X, y, feature_cols = preprocess(df, use_domain_cleaning=True)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)

    # apply SMOTE to training set only
    X_train_s, y_train_s = smote(X_train, np.array(y_train), k=5, random_state=42)

    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss',
        early_stopping_rounds=15, random_state=42,
    )
    model.fit(X_train_s, y_train_s, eval_set=[(X_val, y_val)], verbose=False)
    return model, scaler, feature_cols


if __name__ == '__main__':
    model, scaler, feature_cols = train_model()
    print(f"Trained on {len(feature_cols)} features: {feature_cols}")