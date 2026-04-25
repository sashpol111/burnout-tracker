import streamlit as st
import joblib
import numpy as np
import sys
import time
sys.path.insert(0, '.')
from src.data_loader import load_data, preprocess, split_and_scale
from src.llm_advisor import get_burnout_advice, get_burnout_chat_response

# Load models
@st.cache_resource
def load_models():
    import os
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    
    os.makedirs('models', exist_ok=True)
    
    df = load_data()
    X, y, feature_cols = preprocess(df)
    X_train, X_temp, y_train, y_temp = __import__('sklearn.model_selection', fromlist=['train_test_split']).train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = __import__('sklearn.model_selection', fromlist=['train_test_split']).train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss',
        early_stopping_rounds=15, random_state=42
    )
    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    
    return model, scaler, feature_cols

def get_top_risk_factors(user_input, feature_cols, model):
    importance = model.feature_importances_
    user_values = [user_input[f] for f in feature_cols]
    
    # Weight importance by how extreme the value is (normalized 0-10)
    risk_factors = {}
    for i, col in enumerate(feature_cols):
        val = user_values[i]
        imp = importance[i]
        # Higher stress/shouting/lost vacation = bad, higher sleep/flow = good
        bad_features = ['DAILY_STRESS', 'LOST_VACATION', 'DAILY_SHOUTING', 'BMI_RANGE']
        if col in bad_features:
            score = imp * (val / 10)
        else:
            score = imp * (1 - val / 10)
        risk_factors[col] = score
    
    top = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)[:3]
    return {k: f"score {user_input[k]}/10" for k, _ in top}

def main():
    st.set_page_config(page_title="Burnout Tracker", page_icon="🔥", layout="wide")
    
    st.title("🔥 Burnout Risk Tracker")
    st.markdown("*Predict your burnout risk and get personalized advice*")
    
    xgb_model, scaler, feature_cols = load_models()
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'risk_score' not in st.session_state:
        st.session_state.risk_score = None
    if 'advice' not in st.session_state:
        st.session_state.advice = None

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Daily Lifestyle Inputs")
        
        user_input = {}
        user_input['DAILY_STRESS'] = st.slider("Daily Stress Level", 0, 10, 5)
        user_input['SLEEP_HOURS'] = st.slider("Sleep Hours", 0, 10, 7)
        user_input['LOST_VACATION'] = st.slider("Unused Vacation Days", 0, 10, 2)
        user_input['TODO_COMPLETED'] = st.slider("Tasks Completed Today", 0, 10, 5)
        user_input['FLOW'] = st.slider("Flow State at Work", 0, 10, 5)
        user_input['TIME_FOR_PASSION'] = st.slider("Time for Passions", 0, 10, 3)
        user_input['WEEKLY_MEDITATION'] = st.slider("Weekly Meditation Sessions", 0, 10, 2)
        user_input['DAILY_SHOUTING'] = st.slider("Daily Emotional Outbursts", 0, 10, 1)
        user_input['FRUITS_VEGGIES'] = st.slider("Fruits & Veggies Servings", 0, 10, 5)
        user_input['DAILY_STEPS'] = st.slider("Daily Steps (thousands)", 0, 10, 5)
        user_input['SUFFICIENT_INCOME'] = st.slider("Income Sufficiency", 0, 10, 5)
        user_input['SOCIAL_NETWORK'] = st.slider("Social Network Strength", 0, 10, 5)
        user_input['ACHIEVEMENT'] = st.slider("Sense of Achievement", 0, 10, 5)
        user_input['SUPPORTING_OTHERS'] = st.slider("Supporting Others", 0, 10, 5)
        user_input['PLACES_VISITED'] = st.slider("New Places Visited", 0, 10, 3)
        user_input['CORE_CIRCLE'] = st.slider("Close Relationships", 0, 10, 5)
        user_input['PERSONAL_AWARDS'] = st.slider("Personal Awards/Recognition", 0, 10, 3)
        user_input['DONATION'] = st.slider("Charitable Giving", 0, 10, 2)
        user_input['BMI_RANGE'] = st.slider("BMI Range (1=underweight, 4=obese)", 1, 4, 2)
        user_input['LIVE_VISION'] = st.slider("Life Vision Clarity", 0, 10, 5)
        user_input['AGE'] = st.selectbox("Age Range", [0, 1, 2, 3], 
                                          format_func=lambda x: ['<20', '21-35', '36-50', '51+'][x])
        user_input['GENDER'] = st.selectbox("Gender", [0, 1], 
                                             format_func=lambda x: ['Female', 'Male'][x])

        if st.button("🔍 Assess My Burnout Risk", type="primary"):
            input_array = np.array([[user_input[f] for f in feature_cols]])
            input_scaled = scaler.transform(input_array)
            
            # ⏱️ Start timing
            start_time = time.perf_counter()
            
            risk_score = xgb_model.predict_proba(input_scaled)[0][1]
            
            # ⏱️ End timing
            inference_time = time.perf_counter() - start_time
            
            st.session_state.risk_score = risk_score
            st.session_state.inference_time = inference_time
            st.session_state.conversation_history = []
            
            top_risk_factors = get_top_risk_factors(user_input, feature_cols, xgb_model)
            
            with st.spinner("Getting personalized advice from AI coach..."):
                advice = get_burnout_advice(risk_score, top_risk_factors, user_input)
                st.session_state.advice = advice
                st.session_state.risk_context = f"Risk score: {risk_score:.1%}, Top factors: {top_risk_factors}"

    with col2:
        if 'inference_time' in st.session_state:
            st.caption(f"⚡ Model inference time: {st.session_state.inference_time * 1000:.2f} ms")

        if st.session_state.risk_score is not None:
            risk_score = st.session_state.risk_score
            
            # Risk meter
            st.subheader("Your Burnout Risk")
            if risk_score > 0.7:
                st.error(f"🚨 High Risk: {risk_score:.1%}")
            elif risk_score > 0.4:
                st.warning(f"⚠️ Moderate Risk: {risk_score:.1%}")
            else:
                st.success(f"✅ Low Risk: {risk_score:.1%}")
            
            st.progress(float(risk_score))
            
            # AI advice
            st.subheader("AI Coach Advice")
            st.markdown(st.session_state.advice)
            
            st.divider()
            
            # Chat
            st.subheader("💬 Chat with Your AI Coach")
            
            for msg in st.session_state.conversation_history:
                if msg['role'] == 'user':
                    st.chat_message("user").write(msg['content'])
                else:
                    st.chat_message("assistant").write(msg['content'])
            
            if prompt := st.chat_input("Ask your coach anything..."):
                st.chat_message("user").write(prompt)
                with st.spinner("Thinking..."):
                    response, st.session_state.conversation_history = get_burnout_chat_response(
                        st.session_state.conversation_history,
                        prompt,
                        st.session_state.risk_context
                    )
                st.chat_message("assistant").write(response)
                st.rerun()

if __name__ == '__main__':
    main()