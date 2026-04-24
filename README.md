# Burnout Risk Tracker

A machine learning system that predicts burnout risk from daily lifestyle inputs and provides personalized AI coaching advice.

## What it Does

The Burnout Risk Tracker takes a user's daily lifestyle inputs — including stress levels, sleep hours, flow state, vacation usage, and 18 other signals — and predicts their burnout risk using an ensemble of machine learning models. The system then uses a large language model (Llama 3.3 70B via Groq) to generate personalized, actionable advice based on the user's specific risk factors. Users can also chat with an AI burnout coach that maintains conversation history and references their personal risk profile.

## Quick Start

```bash
git clone https://github.com/sashpol111/burnout-tracker.git
cd burnout-tracker
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo

**[Try it here → huggingface.co/spaces/sashpol/burnout-tracker](https://huggingface.co/spaces/sashpol/burnout-tracker)**

## Video Links

- Demo video: [link]
- Technical walkthrough: [link]

## Evaluation

| Model | Test Accuracy | Test F1 | Test AUC |
|-------|-------------|---------|---------|
| Baseline (majority class) | 0.755 | 0.000 | 0.500 |
| XGBoost (Config 1) | 0.974 | 0.946 | 0.997 |
| XGBoost (Config 2, final) | 0.951 | 0.893 | 0.991 |
| XGBoost (Config 3) | 0.934 | 0.687 | 0.972 |
| Neural Network | 0.966 | 0.935 | 0.999 |
| DistilBERT | 0.878 | 0.753 | 0.925 |

**Ablation Study:**

| Feature Set | Test F1 | Test AUC |
|-------------|---------|---------|
| All 22 features | 0.893 | 0.991 |
| No productivity metrics | 0.819 | 0.975 |
| Stress/health signals only | 0.541 | 0.852 |
| No demographics | 0.889 | 0.990 |
| Top 10 features | 0.767 | 0.955 |

**Synthetic Data:**

| Training Data | Test F1 | Test AUC |
|--------------|---------|---------|
| Real data only | 0.893 | 0.991 |
| Real + synthetic (310 entries) | 0.895 | 0.991 |

## Individual Contributions

- Sasha: [your contributions]
- [Partner name]: [their contributions]