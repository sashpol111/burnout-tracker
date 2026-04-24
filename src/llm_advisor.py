import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_burnout_advice(risk_score, top_risk_factors, user_inputs):
    risk_level = "high" if risk_score > 0.7 else "moderate" if risk_score > 0.4 else "low"
    
    factors_str = "\n".join([f"- {k}: {v}" for k, v in top_risk_factors.items()])
    inputs_str = "\n".join([f"- {k}: {v}" for k, v in user_inputs.items()])
    
    prompt = f"""You are a burnout prevention coach. A user has completed a burnout risk assessment.

Risk Score: {risk_score:.1%} ({risk_level} risk)

Their most concerning factors:
{factors_str}

Their full lifestyle data:
{inputs_str}

Please provide:
1. A brief, empathetic summary of their burnout risk (2-3 sentences)
2. Their top 3 specific risk factors and why they matter
3. Three concrete, actionable recommendations personalized to their situation
4. An encouraging closing message

Keep your response warm, specific, and actionable. Avoid generic advice."""

    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500
    )
    
    return response.choices[0].message.content

def get_burnout_chat_response(conversation_history, user_message, risk_context):
    system_prompt = f"""You are a compassionate burnout prevention coach with expertise in workplace wellness.
You are having a conversation with someone who has just received their burnout risk assessment.

Their risk context:
{risk_context}

Be empathetic, specific, and practical. Reference their specific data when relevant.
Keep responses concise (3-5 sentences) unless they ask for more detail."""

    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        system=system_prompt,
        messages=conversation_history,
        max_tokens=500
    )

    reply = response.choices[0].message.content
    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    return reply, conversation_history


if __name__ == '__main__':
    print("Starting test...")
    test_inputs = {
        "DAILY_STRESS": 8,
        "SLEEP_HOURS": 5,
        "LOST_VACATION": 10,
        "TODO_COMPLETED": 2,
        "TIME_FOR_PASSION": 0
    }
    
    test_risk_factors = {
        "DAILY_STRESS (score 8/10)": "Very high stress levels",
        "SLEEP_HOURS (5hrs)": "Below recommended 7-8 hours",
        "LOST_VACATION (10 days)": "Not taking needed rest"
    }
    
    try:
        print("Calling LLM...")
        advice = get_burnout_advice(0.85, test_risk_factors, test_inputs)
        print(advice)
    except Exception as e:
        print(f"Error: {e}")