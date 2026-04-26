---
title: Burnout Tracker
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: "1.44.0"
app_file: app.py
pinned: false
---


# 🔥 Burnout Risk Tracker with AI Coaching

## 📌 What It Does

This project predicts a user's burnout risk based on lifestyle inputs such as stress, sleep, productivity, and social habits, and then provides personalized, AI-generated advice using a large language model. The goal is to move beyond simple prediction and help users take actionable steps to improve their well-being.

## 🚀 Quick Start

```bash
git clone https://github.com/sashpol111/burnout-tracker.git
cd burnout-tracker
pip install -r requirements.txt
streamlit run app/app.py
```

## 🧠 How It Works

1. User inputs lifestyle data (0–10 scale sliders)
2. Data is scaled and passed into an **XGBoost model**
3. Model predicts burnout probability
4. Top contributing factors are identified
5. LLM generates **personalized coaching advice**
6. User can chat with an AI coach for deeper insights

## 📊 Evaluation

**Preprocessing Pipeline Impact:**

| Condition | F1 | AUC |
|-----------|-----|-----|
| Baseline (no preprocessing) | 0.251 | 0.649 |
| + Domain cleaning | 0.250 | 0.649 |
| + SMOTE | 0.367 | 0.651 |
| + Threshold tuning | 0.535 | 0.649 |
| Full pipeline | 0.543 | 0.651 |

**Regularization Impact:**

| Condition | Train AUC | Test AUC | Gap |
|-----------|-----------|----------|-----|
| No regularization | 1.000 | 0.661 | 0.339 |
| L1 + L2 + early stopping | 0.864 | 0.651 | 0.214 |

## 🎥 Video Links

(Add these before submission)

Demo Video: [link here]
Technical Walkthrough: [link here]

## 📁 Project Structure

```
burnout-tracker/
│
├── app.py
├── data/
├── models/
├── notebooks/
│
├── src/
│   ├── ablation.py
│   ├── baseline.py
│   ├── data_loader.py
│   ├── error_analysis.py
│   ├── hyperparameter_tuning.py
│   ├── llm_advisor.py        # AI prompts + chat
│   ├── synthetic_data.py
│   ├── train_distilbert.py
│   ├── train_neural_net.py
│   ├── train_with_synthetic.py
│   ├── train_xgboost.py
│   └── visualize.py
│
├── .env
├── .gitignore
├── app.py
├── ATTRIBUTION.md
├── README.md
├── requirements.txt
└── SETUP.md
```

## ⚠️ Disclaimer

This tool is for informational purposes only and is not a substitute for professional medical or psychological advice.

## 👩‍💻 Individual Contributions
<<<<<<< HEAD
Sasha Polakov:
Katherine Yu:
=======

**Sasha Polakov:**
- Project architecture and system design
- XGBoost model training and regularization
- Neural network design and training (PyTorch)
- LLM integration (Groq API, multi-turn chat)
- Streamlit app development
- Synthetic data generation pipeline
- Ablation study and hyperparameter tuning
- DistilBERT fine-tuning
- Deployment to Hugging Face Spaces
- Error analysis and visualizations

**Katherine Yu:**
- Label leakage diagnosis and fix (burnout index from Maslach dimensions)
- SMOTE implementation from scratch
- Domain cleaning pipeline (adaptive Likert validation)
- Threshold tuning experiment
- Feature engineering (4 composite features)
- Data pipeline (unified dataset from 3 sources)
- Preprocessing experiment documentation
- SETUP.md
- Hyperparameter grid search

>>>>>>> dca8254c1bba7d99a2a2a54b163abf5c0f0e2ea8
GitHub: https://github.com/sashpol111
