import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path='data/Wellbeing_and_lifestyle_data_Kaggle.csv'):
    df = pd.read_csv(path)
    return df

def preprocess(df, use_clipping=False):
    # Drop timestamp
    df = df.drop(columns=['Timestamp'])
    
    # Encode categorical columns
    df['GENDER'] = df['GENDER'].map({'Female': 0, 'Male': 1})
    age_map = {'Less than 20': 0, '21 to 35': 1, '36 to 50': 2, '51 or more': 3}
    df['AGE'] = df['AGE'].map(age_map)
    
    # Keep only numeric columns
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with nulls
    df = df.dropna()
    
    # Create burnout label: bottom 25% of work-life balance = high burnout risk
    threshold = df['WORK_LIFE_BALANCE_SCORE'].quantile(0.25)
    df['BURNOUT_RISK'] = (df['WORK_LIFE_BALANCE_SCORE'] <= threshold).astype(int)
    
    # Features and target
    feature_cols = [c for c in df.columns if c not in ['WORK_LIFE_BALANCE_SCORE', 'BURNOUT_RISK']]
    X = df[feature_cols]
    y = df['BURNOUT_RISK']

    # Optional outlier handling (percentile clipping)
    if use_clipping:
        for col in df.columns:
            if col not in ['AGE', 'GENDER', 'BMI_RANGE']:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)
    
    return X, y, feature_cols

def split_and_scale(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

if __name__ == '__main__':
    df = load_data()
    X, y, feature_cols = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)
    print(f"Burnout rate: {y.mean():.2%}")
    print(f"Features: {feature_cols}")