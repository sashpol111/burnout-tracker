# ⚙️ Setup Guide – Burnout Risk Tracker

Follow these steps to run the Burnout Tracker locally.

---

## 🧰 Prerequisites

- Python 3.9+
- pip (or conda)
- Git (optional)

---

## 📥 1. Clone the Repository

```bash
git clone https://github.com/sashpol111/burnout-tracker.git
cd burnout-tracker
```

## 🧪 2. Create a Virtual Environment
### macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

## 📦 3. Install Dependencies
```bash
pip install -r requirements.txt
```
If requirements.txt does not work:
```bash
pip install streamlit numpy pandas scikit-learn xgboost python-dotenv groq
```

## 🔐 4. Set Environment Variables
Create a .env file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
```
Get your API key from: https://console.groq.com/

## 🔐 🚀 5. Run the App
Create a .env file in the root directory:
```bash
streamlit run app.py
```
Open in your browser:
http://localhost:8501

