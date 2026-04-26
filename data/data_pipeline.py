import os, json, time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

os.makedirs('data',   exist_ok=True)
os.makedirs('models', exist_ok=True)

# all three sources normalised to these columns before merging
FEATURE_COLS = [
    'FRUITS_VEGGIES', 'PLACES_VISITED', 'CORE_CIRCLE', 'SUPPORTING_OTHERS',
    'SOCIAL_NETWORK', 'ACHIEVEMENT', 'DONATION', 'BMI_RANGE', 'TODO_COMPLETED',
    'FLOW', 'DAILY_STEPS', 'LIVE_VISION', 'SLEEP_HOURS', 'SUFFICIENT_INCOME',
    'PERSONAL_AWARDS', 'TIME_FOR_PASSION', 'WEEKLY_MEDITATION', 'AGE', 'GENDER',
]
TARGET_COL = 'BURNOUT_RISK'

LLAMA_FEATURE_PROMPT = '\n'.join([
    '  "SLEEP_HOURS": 0-10 scale (0=no sleep, 10=excellent sleep)',
    '  "WEEKLY_MEDITATION": 0-10 (0=never, 10=daily)',
    '  "TIME_FOR_PASSION": 0-10 (0=no hobbies, 10=lots of time)',
    '  "TODO_COMPLETED": 0-10 (0=nothing done, 10=very productive)',
    '  "FLOW": 0-10 (0=never in flow, 10=always)',
    '  "ACHIEVEMENT": 0-10 (0=no fulfilment, 10=very fulfilled)',
    '  "LIVE_VISION": 0-10 (0=no direction, 10=clear vision)',
    '  "SOCIAL_NETWORK": 1-10 (1=isolated, 10=strong network)',
    '  "CORE_CIRCLE": 1-10 (1=no close friends, 10=many)',
    '  "SUPPORTING_OTHERS": 0-10',
    '  "FRUITS_VEGGIES": 0-10 (0=poor diet, 10=excellent)',
    '  "DAILY_STEPS": 1-10 (1=sedentary, 10=very active)',
    '  "SUFFICIENT_INCOME": 1-10',
    '  "BMI_RANGE": 1-4 (1=under, 2=normal, 3=over, 4=obese)',
    '  "PERSONAL_AWARDS": 1-10',
    '  "DONATION": 0-10',
    '  "PLACES_VISITED": 0-10',
    '  "AGE": 0-3 (0=under20, 1=21-35, 2=36-50, 3=51+)',
    '  "GENDER": 0=Female 1=Male (use 0 if unknown)',
])

BURNOUT_SYMPTOM_COLS = ['DAILY_STRESS', 'DAILY_SHOUTING', 'LOST_VACATION']

# SOURCE 1: KAGGLE

def load_kaggle_source():
    csv_path = 'data/Wellbeing_and_lifestyle_data_Kaggle.csv'

    if not os.path.exists(csv_path):
        try:
            import kagglehub, glob, shutil
            path = kagglehub.dataset_download('ydalat/lifestyle-and-wellbeing-data')
            csvs = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
            if not csvs:
                raise FileNotFoundError("no CSV in downloaded dataset")
            shutil.copy(csvs[0], csv_path)
        except Exception as e:
            raise RuntimeError(
                f"Kaggle download failed: {e}\n"
                "Place CSV manually in data/ or set KAGGLE_USERNAME and KAGGLE_KEY in .env"
            )

    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Timestamp'], errors='ignore')
    df['GENDER'] = df['GENDER'].map({'Female': 0, 'Male': 1})
    df['AGE']    = df['AGE'].map({'Less than 20': 0, '21 to 35': 1,
                                   '36 to 50': 2,   '51 or more': 3})
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    burnout_index  = df[BURNOUT_SYMPTOM_COLS].sum(axis=1)
    df[TARGET_COL] = (burnout_index >= burnout_index.quantile(0.70)).astype(int)

    available = [c for c in FEATURE_COLS if c in df.columns]
    df        = df[available + [TARGET_COL]].copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 5  # fill missing features with midpoint

    df['source'] = 'kaggle'
    print(f"  Kaggle: {len(df)} rows | burnout rate: {df[TARGET_COL].mean():.1%}")
    return df[FEATURE_COLS + [TARGET_COL, 'source']]

# SOURCE 2: SYNTHETIC MATERIAL

def generate_synthetic_source(n_high=50, n_low=50):
    print(f"Generating {n_high} high-risk + {n_low} low-risk synthetic profiles...")

    def call_llm(risk_level, batch_size=10):
        guidance = (
            "high DAILY_STRESS 7-10, low SLEEP_HOURS 3-6, high LOST_VACATION 7-10, "
            "low FLOW 0-3, low TIME_FOR_PASSION 0-3, high DAILY_SHOUTING 6-10"
            if risk_level == 'high' else
            "low DAILY_STRESS 0-4, high SLEEP_HOURS 7-9, low LOST_VACATION 0-3, "
            "high FLOW 6-10, high TIME_FOR_PASSION 6-10, low DAILY_SHOUTING 0-2"
        )
        all_cols = FEATURE_COLS + ['DAILY_STRESS', 'LOST_VACATION', 'DAILY_SHOUTING']
        prompt = f"""Generate {batch_size} realistic daily lifestyle profiles for people
at {risk_level.upper()} burnout risk. Return ONLY a JSON array with exactly {batch_size} objects.
Each object must have these keys with integer values 0-10 (BMI_RANGE: 1-4, AGE: 0-3, GENDER: 0-1):
{', '.join(all_cols)}

Typical {risk_level} burnout pattern: {guidance}
Return ONLY the JSON array, no explanation, no markdown."""

        response = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {'role': 'system', 'content': 'Return only valid JSON arrays. No markdown.'},
                {'role': 'user',   'content': prompt},
            ],
            max_tokens=3000, temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace('```json', '').replace('```', '').strip()
        return json.loads(raw)

    rows = []
    for risk, n in [('high', n_high), ('low', n_low)]:
        label, generated = (1 if risk == 'high' else 0), 0
        while generated < n:
            batch = min(10, n - generated)
            try:
                data = call_llm(risk, batch_size=batch)
                for entry in data:
                    row = {col: int(np.clip(entry.get(col, 5), 0, 10)) for col in FEATURE_COLS}
                    row['BMI_RANGE'] = int(np.clip(entry.get('BMI_RANGE', 2), 1, 4))
                    row['AGE']       = int(np.clip(entry.get('AGE', 1), 0, 3))
                    row['GENDER']    = int(np.clip(entry.get('GENDER', 0), 0, 1))
                    row[TARGET_COL]  = label
                    row['source']    = 'synthetic'
                    rows.append(row)
                generated += len(data)
                print(f"  {risk} risk: {generated}/{n}")
                time.sleep(0.8)
            except Exception as e:
                print(f"  batch error ({e}), retrying...")
                time.sleep(2)

    df = pd.DataFrame(rows)[FEATURE_COLS + [TARGET_COL, 'source']]
    print(f"  Synthetic: {len(df)} rows | burnout rate: {df[TARGET_COL].mean():.1%}")
    return df

# SOURCE 3: HUGGINGFACE REDDIT POSTS

def load_huggingface_source(max_posts=80):
    print("Loading HuggingFace dataset (solomonk/reddit_mental_health_posts)...")
    try:
        from datasets import load_dataset
        hf_df = load_dataset('solomonk/reddit_mental_health_posts', split='train').to_pandas()
    except Exception as e:
        print(f"  HuggingFace load failed: {e} — skipping")
        return pd.DataFrame()

    text_col  = next((c for c in ['text', 'selftext', 'body'] if c in hf_df.columns), hf_df.columns[0])
    title_col = 'title' if 'title' in hf_df.columns else None
    hf_df['full_text'] = (
        (hf_df[title_col].fillna('') + '\n\n') if title_col else ''
    ) + hf_df[text_col].fillna('')
    hf_df = hf_df[hf_df['full_text'].str.len() >= 150]

    keywords  = ['burnout', 'burn out', 'exhausted', 'overworked', 'stressed',
                 'overwhelmed', 'cant cope', "can't cope", 'work stress']
    kw_mask   = hf_df['full_text'].str.lower().str.contains('|'.join(keywords), na=False)
    n_burnout = min(int(kw_mask.sum()), int(max_posts * 0.6))
    n_other   = min(int((~kw_mask).sum()), max_posts - n_burnout)
    selected  = pd.concat([
        hf_df[kw_mask].sample(n=n_burnout,  random_state=42),
        hf_df[~kw_mask].sample(n=n_other,   random_state=42),
    ]).reset_index(drop=True)

    print(f"  Selected {len(selected)} posts for annotation...")

    def extract_features(text):
        prompt = f"""Analyze this mental health post. Estimate wellness scores.
Post: \"\"\"{text[:1000]}\"\"\"

Estimate:
{LLAMA_FEATURE_PROMPT}

Respond ONLY with valid JSON. No markdown."""
        for _ in range(2):
            try:
                resp = groq_client.chat.completions.create(
                    model='llama-3.3-70b-versatile',
                    messages=[
                        {'role': 'system', 'content': 'Return only valid JSON. No markdown.'},
                        {'role': 'user',   'content': prompt},
                    ],
                    max_tokens=400, temperature=0.1,
                )
                raw  = resp.choices[0].message.content.strip().replace('```json', '').replace('```', '').strip()
                feat = json.loads(raw)
                bounds = {
                    'BMI_RANGE': (1, 4), 'AGE': (0, 3), 'GENDER': (0, 1),
                    'SOCIAL_NETWORK': (1, 10), 'CORE_CIRCLE': (1, 10),
                    'DAILY_STEPS': (1, 10), 'SUFFICIENT_INCOME': (1, 10),
                    'PERSONAL_AWARDS': (1, 10),
                }
                return {col: float(np.clip(feat.get(col, sum(bounds.get(col, (0,10)))/2),
                                           *bounds.get(col, (0, 10))))
                        for col in FEATURE_COLS}
            except Exception:
                time.sleep(1)
        return None

    rows, failed = [], 0
    for _, row in selected.iterrows():
        feat = extract_features(str(row['full_text']))
        if feat is None:
            failed += 1
            continue
        # infer label: high burnout if low sleep + low flow + low achievement
        burnout_score    = (10 - feat['SLEEP_HOURS']) + (10 - feat['FLOW']) + (10 - feat['ACHIEVEMENT'])
        feat[TARGET_COL] = 1 if burnout_score > 18 else 0
        feat['source']   = 'huggingface'
        rows.append(feat)
        if len(rows) % 20 == 0:
            print(f"  Annotated {len(rows)}/{len(selected)}...")
        time.sleep(0.3)

    df = pd.DataFrame(rows)[FEATURE_COLS + [TARGET_COL, 'source']]
    print(f"  HuggingFace: {len(df)} rows | burnout rate: {df[TARGET_COL].mean():.1%} | {failed} failed")
    return df


def build_unified_dataset(
    use_kaggle=True, use_synthetic=True, use_huggingface=True,
    n_synthetic_high=50, n_synthetic_low=50, n_hf_posts=80,
    save_path='data/unified_dataset.csv', force_rebuild=False,
):
    if not force_rebuild and os.path.exists(save_path):
        df = pd.read_csv(save_path)
        print(f"loaded {save_path}: {len(df)} rows | {df['source'].value_counts().to_dict()}")
        return df

    parts = []
    for flag, label, fn, args in [
        (use_kaggle,      'kaggle',      load_kaggle_source,       []),
        (use_synthetic,   'synthetic',   generate_synthetic_source, [n_synthetic_high, n_synthetic_low]),
        (use_huggingface, 'huggingface', load_huggingface_source,  [n_hf_posts]),
    ]:
        if not flag:
            continue
        try:
            result = fn(*args)
            if len(result) > 0:
                parts.append(result)
        except Exception as e:
            print(f"  {label} source failed: {e}")

    if not parts:
        raise RuntimeError("all data sources failed — check API keys and connectivity")

    unified = pd.concat(parts, ignore_index=True).dropna(subset=FEATURE_COLS)
    for col in FEATURE_COLS + [TARGET_COL]:
        unified[col] = pd.to_numeric(unified[col], errors='coerce').fillna(0)

    unified.to_csv(save_path, index=False)
    print(f"saved {save_path}: {len(unified)} rows | {unified['source'].value_counts().to_dict()} | "
          f"burnout rate {unified[TARGET_COL].mean():.1%}")
    return unified


if __name__ == '__main__':
    df = build_unified_dataset(
        use_kaggle=True, use_synthetic=True, use_huggingface=True,
        n_synthetic_high=50, n_synthetic_low=50, n_hf_posts=60,
        force_rebuild=True,
    )
    print(df.groupby('source').head(1)[['source', 'SLEEP_HOURS', 'FLOW', 'ACHIEVEMENT', TARGET_COL]].to_string())