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
Model used: XGBoost classifier
Metric: Probability of burnout risk
Inference time: ~20–30 ms

Example result:

Risk Score: 22% (Low Risk)
AI-generated recommendations tailored to user inputs

Future improvements:

Add accuracy metrics (precision, recall, F1)
Compare multiple models
Perform deeper error analysis

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

## 👩‍💻 Authors
Sasha Polakov
Katherine Yu
GitHub: https://github.com/sashpol111