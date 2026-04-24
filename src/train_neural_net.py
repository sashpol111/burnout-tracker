import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, split_and_scale

class BurnoutNet(nn.Module):
    def __init__(self, input_dim):
        super(BurnoutNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_proba, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            proba = model(X_batch).cpu().numpy()
            preds = (proba >= 0.5).astype(int)
            all_proba.extend(proba)
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_proba)
    return acc, f1, auc

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = load_data()
    X, y, feature_cols = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train.values)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val.values)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test.values)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64)

    model = BurnoutNet(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    # Training loop
    best_val_auc = 0
    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(100):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        val_acc, val_f1, val_auc = evaluate(model, val_loader, device)
        scheduler.step(1 - val_auc)
        train_losses.append(train_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f} | Val AUC: {val_auc:.3f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'models/best_neural_net.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load('models/best_neural_net.pt'))
    train_acc, train_f1, train_auc = evaluate(model, train_loader, device)
    val_acc, val_f1, val_auc = evaluate(model, val_loader, device)
    test_acc, test_f1, test_auc = evaluate(model, test_loader, device)

    print(f"\n=== Neural Network Results ===")
    print(f"Train — Accuracy: {train_acc:.3f} | F1: {train_f1:.3f} | AUC: {train_auc:.3f}")
    print(f"Val   — Accuracy: {val_acc:.3f} | F1: {val_f1:.3f} | AUC: {val_auc:.3f}")
    print(f"Test  — Accuracy: {test_acc:.3f} | F1: {test_f1:.3f} | AUC: {test_auc:.3f}")

    joblib.dump(train_losses, 'models/train_losses.pkl')
    print("\nModel saved to models/best_neural_net.pt")