# Setup Guide — Burnout Risk Tracker

## Prerequisites

- Python 3.9+
- pip or conda

---

## 1. Clone the Repository

```bash
git clone https://github.com/sashpol111/burnout-tracker.git
cd burnout-tracker
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. API Keys

This project requires two external API keys.

**Groq API** (required for LLM coaching and chat):
1. Go to https://console.groq.com and create a free account
2. Generate an API key from the dashboard

**Kaggle API** (required only if rebuilding the dataset from scratch — not needed to run the app):
1. Go to https://www.kaggle.com/settings/account
2. Under the API section, click "Generate New Token"
3. Copy the `username` and `key` values from that file

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

---

## 4. Run the App

```bash
streamlit run app.py
```

Open your browser to http://localhost:8501

---

## 5. Rebuilding the Dataset (optional)

The app ships with `data/unified_dataset.csv` already built. If you want to regenerate it from all three API sources, run:

```bash
python data/data_pipeline.py
```

This will call the Kaggle API, Groq API, and HuggingFace Datasets API and may take several minutes.

---

## Notes for Graders

The app runs fully offline after setup — the only live API call at runtime is to Groq for the coaching advice and chat responses. If you do not have a Groq API key, the risk prediction and sliders will still work, but the AI coach section will show an error message. All ML experiments/evidence can be run independently without launching the app:

```bash
python src/preprocessing_experiment.py
python src/hyperparameter_tuning.py
python src/error_analysis.py
python src/ablation.py
```