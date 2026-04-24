import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess

def row_to_text(row, feature_cols):
    """Convert a row of lifestyle data to a natural language description."""
    text = (
        f"Daily stress level {row['DAILY_STRESS']:.0f} out of 10. "
        f"Sleeps {row['SLEEP_HOURS']:.0f} hours per night. "
        f"Has {row['LOST_VACATION']:.0f} unused vacation days. "
        f"Flow state at work {row['FLOW']:.0f} out of 10. "
        f"Time for passion {row['TIME_FOR_PASSION']:.0f} out of 10. "
        f"Weekly meditation sessions {row['WEEKLY_MEDITATION']:.0f}. "
        f"Daily shouting or emotional outbursts {row['DAILY_SHOUTING']:.0f}. "
        f"Tasks completed {row['TODO_COMPLETED']:.0f} out of 10. "
        f"Sense of achievement {row['ACHIEVEMENT']:.0f} out of 10. "
        f"Social network strength {row['SOCIAL_NETWORK']:.0f} out of 10."
    )
    return text

class BurnoutTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, 
                                   max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_proba, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            proba = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = (proba >= 0.5).astype(int)
            
            all_proba.extend(proba)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_proba)
    return acc, f1, auc

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    df = load_data()
    X, y, feature_cols = preprocess(df)
    
    # Reconstruct original df with feature names for text generation
    df_processed = pd.DataFrame(X, columns=feature_cols)
    df_processed['BURNOUT_RISK'] = y.values
    
    # Convert to text
    print("Converting data to text descriptions...")
    texts = [row_to_text(row, feature_cols) for _, row in df_processed.iterrows()]
    labels = df_processed['BURNOUT_RISK'].values.tolist()
    
    # Use subset for speed - 3000 samples
    idx = np.random.choice(len(texts), 3000, replace=False)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Tokenize
    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = BurnoutTextDataset(X_train, y_train, tokenizer)
    val_dataset = BurnoutTextDataset(X_val, y_val, tokenizer)
    test_dataset = BurnoutTextDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load model
    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training
    best_val_f1 = 0
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        val_acc, val_f1, val_auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f} | Val AUC: {val_auc:.3f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'models/best_distilbert.pt')
    
    # Final evaluation
    model.load_state_dict(torch.load('models/best_distilbert.pt'))
    test_acc, test_f1, test_auc = evaluate(model, test_loader, device)
    print(f"\n=== DistilBERT Results ===")
    print(f"Test Accuracy: {test_acc:.3f} | Test F1: {test_f1:.3f} | Test AUC: {test_auc:.3f}")
    
    print("\n=== Final Model Comparison ===")
    print(f"{'Model':20s} | Test F1 | Test AUC")
    print("-" * 45)
    print(f"{'Baseline':20s} | 0.000   | 0.500")
    print(f"{'XGBoost':20s} | 0.893   | 0.991")
    print(f"{'Neural Network':20s} | 0.935   | 0.999")
    print(f"{'DistilBERT':20s} | {test_f1:.3f}   | {test_auc:.3f}")