# 🔥 Burnout Risk Tracker with AI Coaching

Predict your burnout risk and get personalized, AI-powered advice based on your daily lifestyle.

---

## 🚀 Overview

This app helps users understand and prevent burnout by combining:

- ⚡ **Machine Learning (XGBoost)** → predicts burnout risk instantly  
- 🤖 **AI Coaching (LLaMA 3 via Groq)** → provides personalized advice  
- 💬 **Interactive Chat** → follow-up questions with an AI coach  

👉 Instead of just predicting risk, the app helps users **take action to improve their well-being**.

---

## ✨ Features

- 🔍 **Burnout Risk Prediction**
  - Real-time risk score based on lifestyle inputs

- ⚡ **Fast Model Inference**
  - ~20–30 ms per prediction

- 🤖 **Personalized AI Advice**
  - Tailored recommendations based on your inputs

- 💬 **AI Coach Chat**
  - Ask follow-up questions about stress, habits, or burnout

- 📊 **Comprehensive Lifestyle Tracking**
  - Stress, sleep, habits, social life, health, and more

---

## 🧠 How It Works

1. User inputs lifestyle data (0–10 scale sliders)
2. Data is scaled and passed into an **XGBoost model**
3. Model predicts burnout probability
4. Top contributing factors are identified
5. LLM generates **personalized coaching advice**
6. User can chat with an AI coach for deeper insights

---

## 🖥️ App Interface

### Inputs

Users provide daily lifestyle data including:

- Stress level
- Sleep hours
- Work productivity
- Emotional health
- Physical activity
- Social connections
- Life satisfaction metrics

---

### Example Output

**⚡ Model inference time:** 26.41 ms  

**Burnout Risk:**  
✅ Low Risk: 22.0%

---

### 🤖 AI Coach Advice (Example)

> Your risk score is low, which is great — you're already doing many things right.  
> However, a few areas could be improved to maintain long-term well-being.

**Top Risk Factors:**
- Task completion consistency  
- Sense of achievement  
- Lack of new experiences  

**Recommendations:**
- Break tasks into smaller steps to improve completion  
- Set realistic goals and celebrate progress  
- Schedule time for new experiences or exploration  

---

## 🏗️ Tech Stack

### Frontend
- Streamlit

### Machine Learning
- XGBoost
- Scikit-learn

### AI / LLM
- Groq API
- LLaMA 3 (70B)

### Data & Utilities
- NumPy
- Pandas
- python-dotenv

---

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

## ⚙️ Setup

See full setup instructions:

👉 [SETUP.md](./SETUP.md)

### Quick Start
```bash
git clone https://github.com/sashpol111/burnout-tracker.git
cd burnout-tracker
pip install -r requirements.txt
streamlit run app/app.py
```

## ⏱ Performance
Model inference: ~20–30 ms
AI response time: ~1–3 seconds

## ⚠️ Disclaimer

This tool is for informational purposes only and is not a substitute for professional medical or psychological advice.

## 📚 Attribution
XGBoost → https://xgboost.readthedocs.io/
Scikit-learn → https://scikit-learn.org/
Streamlit → https://streamlit.io/
Groq → https://groq.com/

## 👩‍💻 Authors
Sasha Polakov
Katherine Yu
GitHub: https://github.com/sashpol111