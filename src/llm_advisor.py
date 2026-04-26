import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"


def get_burnout_advice(risk_score, top_risk_factors, user_inputs):
    risk_level   = "high" if risk_score > 0.7 else "moderate" if risk_score > 0.4 else "low"
    factors_str  = "\n".join([f"- {k}: {v}" for k, v in top_risk_factors.items()])
    inputs_str   = "\n".join([f"- {k}: {v}" for k, v in user_inputs.items()])

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
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content


def get_burnout_chat_response(conversation_history, user_message, risk_context):
    """
    Multi-turn chat with Llama 3.3 70B via Groq.

    The system prompt is passed as the first element of the messages list —
    this is the correct format for the Groq API (and OpenAI-compatible APIs
    generally). Passing `system=` as a top-level kwarg is NOT supported and
    silently fails, which was the original bug causing the chat to not respond.

    Pipeline position: receives XGBoost risk_context (score + top factors)
    and injects it into every turn so the LLM always has the ML output in view.
    """
    system_prompt = f"""You are a compassionate burnout prevention coach with expertise in workplace wellness.
You are having a conversation with someone who has just received their burnout risk assessment.

Their risk context:
{risk_context}

Be empathetic, specific, and practical. Reference their specific data when relevant.
Keep responses concise (3-5 sentences) unless they ask for more detail."""

    # building full message list
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user",   "content": user_message},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
    )

    reply = response.choices[0].message.content

    # append both turns to history so context grows correctly across turns
    updated_history = conversation_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]

    return reply, updated_history


if __name__ == '__main__':
    print("Testing get_burnout_advice...")
    advice = get_burnout_advice(
        risk_score=0.85,
        top_risk_factors={
            "SLEEP_HOURS": "score 5/10",
            "WEEKLY_MEDITATION": "score 0/10",
            "FLOW": "score 2/10",
        },
        user_inputs={
            "SLEEP_HOURS": 5, "WEEKLY_MEDITATION": 0,
            "TIME_FOR_PASSION": 1, "SOCIAL_NETWORK": 3,
        },
    )
    print(advice)

    print("\nTesting multi-turn chat...")
    history = []
    reply, history = get_burnout_chat_response(
        history, "What should I do first?", "Risk score: 85%, top factor: low sleep"
    )
    print(f"Turn 1: {reply}")

    reply, history = get_burnout_chat_response(
        history, "Can you give me a sleep routine?", "Risk score: 85%, top factor: low sleep"
    )
    print(f"Turn 2: {reply}")
    print(f"History length: {len(history)} messages")