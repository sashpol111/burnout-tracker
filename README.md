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

A machine learning app that predicts burnout risk from lifestyle inputs and delivers personalized coaching through a large language model. Built with XGBoost, Llama 3.3 70B, and Streamlit.

---

## What It Does

Burnout develops gradually from lifestyle patterns, making it hard to catch early. This project predicts burnout risk from upstream wellness behaviors like sleep, meditation, social support, and flow at work. A user enters their daily habits through interactive sliders, and an XGBoost classifier trained on roughly 16,000 wellness survey responses outputs a burnout risk probability. The risk score and top contributing factors go directly into a prompt for Llama 3.3 70B (via Groq API), which generates personalized coaching advice grounded in the user's specific results. The user can then chat with the AI coach, which remembers the full conversation and the original risk assessment. Training data comes from three sources via a multi-API pipeline: the Kaggle wellness survey, LLM-generated synthetic profiles, and annotated mental health posts from HuggingFace.

---

## Quick Start

```bash
git clone https://github.com/sashpol111/burnout-tracker.git
cd burnout-tracker
pip install -r requirements.txt
streamlit run app.py
```

See `SETUP.md` for full installation instructions and API key setup.

---

## Video Links

- **Demo Video:** [link here]
- **Technical Walkthrough:** [link here]

---

## Evaluation

**Dataset:** 16,131 rows built from the Kaggle wellness survey, LLM-generated synthetic profiles, and HuggingFace mental health posts via `data/data_pipeline.py`.

**Target variable:** Top 30% of a burnout symptom composite (DAILY_STRESS + DAILY_SHOUTING + LOST_VACATION), motivated by Maslach's three burnout dimensions. The original WORK_LIFE_BALANCE_SCORE column was excluded after a leakage diagnostic confirmed R²=1.0 with all features.

**Preprocessing results:**

| Condition | F1 | AUC |
|---|---|---|
| Baseline (no interventions) | 0.240 | 0.650 |
| Domain cleaning | 0.252 | 0.649 |
| SMOTE oversampling | 0.366 | 0.653 |
| Threshold tuning | 0.538 | 0.649 |
| Full pipeline | 0.542 | 0.653 |

F1 improved from 0.240 to 0.542, a 125% relative gain. AUC stays flat across all conditions, confirming the interventions shift the operating point rather than changing the model's underlying discriminative ability. Full results in `src/preprocessing_experiment.py`.

**Regularization results:**

| Condition | Train AUC | Test AUC | Gap |
|---|---|---|---|
| No regularization (depth=8) | 1.000 | 0.667 | 0.333 |
| L1 + L2 + early stopping | 0.867 | 0.656 | 0.212 |

The overfitting gap dropped 36% while test AUC stayed comparable, confirming regularization prevents memorization without hurting generalization. Full ablation across all regularization conditions in `src/preprocessing_experiment.py`.

**Hyperparameter tuning:** A 9-configuration grid search over reg_alpha in {0.0, 0.1, 1.0} and reg_lambda in {0.1, 1.0, 5.0} selected reg_alpha=0.1, reg_lambda=5.0 (val AUC=0.679, test AUC=0.656). Both penalties are kept because L1 performs feature selection and L2 shrinks weights smoothly — the combination outperformed either alone across the grid. Full results in `src/hyperparameter_tuning.py`.

**Inference time:** 3-5 ms per prediction, measured via `time.perf_counter()` and displayed in the app UI after each assessment.

**Error analysis:** The model uses a tuned threshold of 0.36 on a test set of 2,420 samples, producing 187 false negatives and 902 false positives. The model prioritizes recall for high risk cases (0.77) over precision (0.42), which fits a burnout screening context where missing a real case is more costly than a false alarm. The hardest feature to classify around is TODO_COMPLETED, which shows the largest gap between missed and caught cases. Borderline predictions within 0.10 of the threshold achieve 53.8% accuracy versus 56.8% for confident predictions, confirming errors concentrate near the decision boundary. Full visualizations in `docs/error_analysis.png`.

---

## Project Structure

```
burnout-tracker/
├── app.py                          # Streamlit application
├── data/
│   ├── data_pipeline.py            # Multi-API data collection pipeline (Kaggle 
                                    +  synthetic + HF)
│   └── unified_dataset.csv         # Merged dataset created by the pipeline 
├── src/
│   ├── data_loader.py              # Preprocessing, feature engineering, splitting
│   ├── smote.py                    # Custom SMOTE (Chawla et al. 2002)
│   ├── hyperparameter_tuning.py    # Grid search over regularization params
│   ├── preprocessing_experiment.py # Preprocessing and regularization experiments
│   ├── ablation.py                 # Feature group ablation study
│   ├── error_analysis.py           # Error analysis with visualizations
│   ├── llm_advisor.py              # Groq API integration for advice and chat
├── docs/
│   ├── error_analysis.png
├── videos/
│   └── README.md                   # Links to demo and walkthrough videos
├── ATTRIBUTION.md
├── README.md
├── SETUP.md
├── requirements.txt
```

---

## Individual Contributions

**Sasha Polakov:**
[describe contributions]

**Katherine Yu:**
Redesigned the target variable construction after identifying circular leakage, implemented custom SMOTE, and developed the preprocessing experiment pipeline. Built the three-source unified dataset pipeline integrating Kaggle, Groq, and HuggingFace APIs. Applied feature engineering to construct domain-motivated composite features. Reworked LLM integration to fix multi-turn context management and corrected errors in regularization and feature group ablation. Coded hyperparameter grid search and error analysis.


GitHub: https://github.com/sashpol111/burnout-tracker

---

## Disclaimer

This tool is for informational purposes only and is not a substitute for professional medical or psychological advice.