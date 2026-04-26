import streamlit as st
import numpy as np
import sys
import time
sys.path.insert(0, '.')
from data.data_loader import load_data, preprocess, split_and_scale
from src.llm_advisor import get_burnout_advice, get_burnout_chat_response


@st.cache_resource
def load_models():
    from src.train_xgboost import train_model
    return train_model()


def get_top_risk_factors(user_input, feature_cols, model):
    """Return top-3 risk factors using XGBoost feature importances."""
    importances = model.feature_importances_
    scores = {}
    for col, imp in zip(feature_cols, importances):
        if col not in user_input:
            continue
        val = user_input[col]
        # Recovery/wellness features: low value = higher risk contribution
        recovery_features = [
            'SLEEP_HOURS', 'WEEKLY_MEDITATION', 'TIME_FOR_PASSION', 'FLOW',
            'ACHIEVEMENT', 'SOCIAL_NETWORK', 'CORE_CIRCLE', 'LIVE_VISION',
            'RECOVERY_SCORE', 'SOCIAL_SUPPORT_SCORE', 'LIFESTYLE_SCORE', 'HEALTH_HABITS',
        ]
        if col in recovery_features:
            scores[col] = imp * (1 - val / 10)
        else:
            scores[col] = imp * (val / 10)

    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return {k: f"score {user_input.get(k, '?')}/10" for k, _ in top3}


def main():
    st.set_page_config(page_title="Burnout Risk Tracker", page_icon="🔥", layout="wide")

    st.title("🔥 Burnout Risk Tracker")
    st.markdown(
        "*Answer questions about your **recovery and lifestyle habits** — "
        "the model predicts whether you're showing early burnout risk.*"
    )

    xgb_model, scaler, feature_cols = load_models()

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'risk_score' not in st.session_state:
        st.session_state.risk_score = None
    if 'advice' not in st.session_state:
        st.session_state.advice = None

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Your Lifestyle & Recovery Habits")
        st.caption(
            "These questions cover your wellness behaviors — not your current stress level. "
            "The model infers burnout risk from how you're living, not how you're feeling right now."
        )

        user_input = {}

        st.markdown("**Sleep & Recovery**")
        user_input['SLEEP_HOURS']        = st.slider("Sleep hours per night", 0, 10, 7)
        user_input['WEEKLY_MEDITATION']  = st.slider("Meditation sessions per week", 0, 10, 2)
        user_input['TIME_FOR_PASSION']   = st.slider("Time for hobbies / passions", 0, 10, 3)

        st.markdown("**Work & Productivity**")
        user_input['TODO_COMPLETED']     = st.slider("Daily tasks completed", 0, 10, 5)
        user_input['FLOW']               = st.slider("Flow state at work", 0, 10, 5)
        user_input['ACHIEVEMENT']        = st.slider("Sense of achievement", 0, 10, 5)
        user_input['LIVE_VISION']        = st.slider("Clarity of life vision", 0, 10, 5)

        st.markdown("**Social & Support**")
        user_input['SOCIAL_NETWORK']     = st.slider("Strength of social network", 0, 10, 5)
        user_input['CORE_CIRCLE']        = st.slider("Close / trusted relationships", 0, 10, 5)
        user_input['SUPPORTING_OTHERS']  = st.slider("Supporting others regularly", 0, 10, 5)

        st.markdown("**Health & Lifestyle**")
        user_input['FRUITS_VEGGIES']     = st.slider("Fruit & veg servings per day", 0, 10, 5)
        user_input['DAILY_STEPS']        = st.slider("Daily steps (thousands)", 0, 10, 5)
        user_input['SUFFICIENT_INCOME']  = st.slider("Income sufficiency", 0, 10, 5)
        user_input['BMI_RANGE']          = st.slider("BMI range (1=under, 4=obese)", 1, 4, 2)

        st.markdown("**Personal Growth**")
        user_input['PERSONAL_AWARDS']    = st.slider("Personal awards / recognition", 0, 10, 3)
        user_input['DONATION']           = st.slider("Charitable giving", 0, 10, 2)
        user_input['PLACES_VISITED']     = st.slider("New places visited recently", 0, 10, 3)

        st.markdown("**About you**")
        user_input['AGE']    = st.selectbox(
            "Age range", [0, 1, 2, 3],
            format_func=lambda x: ['Under 20', '21–35', '36–50', '51+'][x]
        )
        user_input['GENDER'] = st.selectbox(
            "Gender", [0, 1],
            format_func=lambda x: ['Female', 'Male'][x]
        )

        # Fill in engineered features so the input vector is complete
        user_input['RECOVERY_SCORE']       = (user_input['SLEEP_HOURS']
                                               + user_input['TIME_FOR_PASSION']
                                               + user_input['WEEKLY_MEDITATION'])
        user_input['SOCIAL_SUPPORT_SCORE'] = (user_input['SOCIAL_NETWORK']
                                               + user_input['CORE_CIRCLE'])
        user_input['LIFESTYLE_SCORE']      = (user_input['FLOW']
                                               + user_input['ACHIEVEMENT']
                                               + user_input['LIVE_VISION']
                                               + user_input['TIME_FOR_PASSION'])
        user_input['HEALTH_HABITS']        = (user_input['FRUITS_VEGGIES']
                                               + user_input['SLEEP_HOURS']
                                               + user_input['TODO_COMPLETED'])

        if st.button("🔍 Assess My Burnout Risk", type="primary"):
            try:
                input_array  = np.array([[user_input.get(f, 0) for f in feature_cols]])
                input_scaled = scaler.transform(input_array)

                t0         = time.perf_counter()
                risk_score = xgb_model.predict_proba(input_scaled)[0][1]
                inference_ms = (time.perf_counter() - t0) * 1000

                st.session_state.risk_score          = risk_score
                st.session_state.inference_time_ms   = inference_ms
                st.session_state.conversation_history = []

                top_risk_factors = get_top_risk_factors(user_input, feature_cols, xgb_model)

                with st.spinner("Getting personalised advice from AI coach..."):
                    advice = get_burnout_advice(risk_score, top_risk_factors, user_input)
                    st.session_state.advice       = advice
                    st.session_state.risk_context = (
                        f"Risk score: {risk_score:.1%}, "
                        f"Top factors: {top_risk_factors}"
                    )
            except Exception as e:
                st.error(f"Something went wrong during prediction: {e}")

    with col2:
        if st.session_state.risk_score is not None:
            risk_score = st.session_state.risk_score

            st.caption(f"⚡ Model inference: {st.session_state.inference_time_ms:.2f} ms")

            st.subheader("Your Burnout Risk")
            if risk_score > 0.7:
                st.error(f"🚨 High Risk: {risk_score:.1%}")
            elif risk_score > 0.4:
                st.warning(f"⚠️ Moderate Risk: {risk_score:.1%}")
            else:
                st.success(f"✅ Low Risk: {risk_score:.1%}")

            st.progress(float(risk_score))

            st.subheader("AI Coach Advice")
            st.markdown(st.session_state.advice)

            st.divider()

            st.subheader("💬 Chat with Your AI Coach")

            for msg in st.session_state.conversation_history:
                role = msg['role']
                st.chat_message(role).write(msg['content'])

            if prompt := st.chat_input("Ask your coach anything..."):
                st.chat_message("user").write(prompt)
                with st.spinner("Thinking..."):
                    try:
                        response, st.session_state.conversation_history = get_burnout_chat_response(
                            st.session_state.conversation_history,
                            prompt,
                            st.session_state.risk_context,
                        )
                    except Exception as e:
                        response = f"Sorry, I couldn't reach the AI coach right now ({e})."
                st.chat_message("assistant").write(response)
                st.rerun()


if __name__ == '__main__':
    main()