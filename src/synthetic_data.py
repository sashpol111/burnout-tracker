import os
import pandas as pd
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import json
import time

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_burnout_entries(n_high=100, n_low=100):
    """Generate synthetic daily lifestyle entries labeled by burnout risk."""
    
    entries = []
    
    # Generate HIGH burnout risk profiles
    print(f"Generating {n_high} high burnout profiles...")
    for i in range(0, n_high, 5):
        prompt = """Generate 10 realistic daily lifestyle profiles for people at HIGH burnout risk.
Return ONLY a JSON array with exactly 10 objects. Each object must have these exact keys with integer values 0-10 (BMI_RANGE: 1-4):
FRUITS_VEGGIES, DAILY_STRESS, PLACES_VISITED, CORE_CIRCLE, SUPPORTING_OTHERS, SOCIAL_NETWORK, 
ACHIEVEMENT, DONATION, BMI_RANGE, TODO_COMPLETED, FLOW, DAILY_STEPS, LIVE_VISION, SLEEP_HOURS, 
LOST_VACATION, DAILY_SHOUTING, SUFFICIENT_INCOME, PERSONAL_AWARDS, TIME_FOR_PASSION, 
WEEKLY_MEDITATION, AGE, GENDER

High burnout profiles typically have: high DAILY_STRESS (7-10), low SLEEP_HOURS (3-6), 
high LOST_VACATION (7-10), low FLOW (0-3), low TIME_FOR_PASSION (0-3), high DAILY_SHOUTING (6-10).

Return ONLY the JSON array, no other text."""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000
            )
            text = response.choices[0].message.content.strip()
            # Clean up response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            for entry in data:
                entry['BURNOUT_RISK'] = 1
                entries.append(entry)
            print(f"  Generated {min(i+10, n_high)}/{n_high} high risk profiles")
            time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Generate LOW burnout risk profiles
    print(f"Generating {n_low} low burnout profiles...")
    for i in range(0, n_low, 5):
        prompt = """Generate 10 realistic daily lifestyle profiles for people at LOW burnout risk.
Return ONLY a JSON array with exactly 10 objects. Each object must have these exact keys with integer values 0-10 (BMI_RANGE: 1-4):
FRUITS_VEGGIES, DAILY_STRESS, PLACES_VISITED, CORE_CIRCLE, SUPPORTING_OTHERS, SOCIAL_NETWORK, 
ACHIEVEMENT, DONATION, BMI_RANGE, TODO_COMPLETED, FLOW, DAILY_STEPS, LIVE_VISION, SLEEP_HOURS, 
LOST_VACATION, DAILY_SHOUTING, SUFFICIENT_INCOME, PERSONAL_AWARDS, TIME_FOR_PASSION, 
WEEKLY_MEDITATION, AGE, GENDER

Low burnout profiles typically have: low DAILY_STRESS (0-4), high SLEEP_HOURS (7-9), 
low LOST_VACATION (0-3), high FLOW (6-10), high TIME_FOR_PASSION (6-10), low DAILY_SHOUTING (0-2).

Return ONLY the JSON array, no other text."""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            for entry in data:
                entry['BURNOUT_RISK'] = 0
                entries.append(entry)
            print(f"  Generated {min(i+10, n_low)}/{n_low} low risk profiles")
            time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    df = pd.DataFrame(entries)
    import os
    if os.path.exists('data/synthetic_burnout_data.csv'):
        existing = pd.read_csv('data/synthetic_burnout_data.csv')
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv('data/synthetic_burnout_data.csv', index=False)
    print(f"\nSaved {len(df)} synthetic entries to data/synthetic_burnout_data.csv")
    return df

if __name__ == '__main__':
    df = generate_burnout_entries(n_high=0, n_low=55)
    print(f"Total entries now: {len(df)}")
    print(f"Burnout rate: {df['BURNOUT_RISK'].mean():.2%}")