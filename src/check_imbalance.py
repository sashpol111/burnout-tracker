from data_loader import load_data, preprocess

df = load_data()
X, y, _ = preprocess(df)

print("Class distribution:")
print(y.value_counts())
print(f"Burnout rate: {y.mean():.2%}")