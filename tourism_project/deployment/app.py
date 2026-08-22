
"""
Streamlit front-end for the Tourism Package Prediction model.
Collects customer attributes through a styled, well-organized form,
builds a single-row feature frame matching the trained model's schema,
and surfaces the predicted purchase outcome with a visual probability
gauge and a friendly summary of the customer profile.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------------------
# Page setup (must be the first Streamlit call)
# --------------------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------
# Custom styling
# --------------------------------------------------------------

st.markdown(
    """
    <style>
        /* Overall page */
        .stApp {
            background: linear-gradient(180deg, #f7fafc 0%, #eef2f7 100%);
        }

        /* Hero banner */
        .hero-banner {
            background: linear-gradient(120deg, #0f766e 0%, #0891b2 50%, #2563eb 100%);
            padding: 2.2rem 2rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px rgba(15, 118, 110, 0.25);
        }
        .hero-banner h1 {
            font-size: 2.1rem;
            margin-bottom: 0.3rem;
            color: white;
        }
        .hero-banner p {
            font-size: 1.02rem;
            opacity: 0.92;
            margin: 0;
        }

        /* Section card headers */
        .section-header {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f766e;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.4rem;
            margin-bottom: 0.8rem;
            margin-top: 0.4rem;
        }

        /* Result cards */
        .result-card-positive {
            background: linear-gradient(120deg, #ecfdf5, #d1fae5);
            border-left: 6px solid #059669;
            padding: 1.4rem 1.6rem;
            border-radius: 14px;
            margin-top: 0.5rem;
        }
        .result-card-negative {
            background: linear-gradient(120deg, #fff7ed, #ffedd5);
            border-left: 6px solid #ea580c;
            padding: 1.4rem 1.6rem;
            border-radius: 14px;
            margin-top: 0.5rem;
        }
        .result-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .result-subtitle {
            font-size: 0.95rem;
            color: #475569;
        }

        /* Sidebar tweaks */
        section[data-testid="stSidebar"] {
            background: #0f172a;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        /* Metric styling */
        div[data-testid="stMetric"] {
            background: white;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }

        /* Predict button */
        div.stButton > button {
            background: linear-gradient(120deg, #0f766e, #2563eb);
            color: white;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 0.7rem 1rem;
            border-radius: 12px;
            border: none;
            width: 100%;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.35);
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------
# Model loading
# --------------------------------------------------------------

@st.cache_resource
def load_trained_model():
    model_file = Path(__file__).resolve().parent / "tourism_model.pkl"
    return joblib.load(model_file)


predictor = load_trained_model()


# --------------------------------------------------------------
# Hero banner
# --------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
        <h1>🧳 Tourism Package Prediction</h1>
        <p>Fill in the customer's profile and let the model estimate their
        likelihood of purchasing a tourism package — instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------
# Sidebar — quick guide
# --------------------------------------------------------------

with st.sidebar:
    st.markdown("## ℹ️ How it works")
    st.write(
        "1. Enter the customer's details in the form.\n"
        "2. Click **Predict Package Purchase**.\n"
        "3. Review the prediction, confidence score, and full profile."
    )
    st.markdown("---")
    st.markdown("## 🧭 About")
    st.write(
        "This tool uses a trained machine learning model to estimate "
        "whether a customer is likely to buy a tourism package, based "
        "on demographic, behavioral, and sales-interaction data."
    )
    st.markdown("---")
    st.caption("Built with Streamlit • Powered by scikit-learn")


# --------------------------------------------------------------
# Input form
# --------------------------------------------------------------

st.markdown('<div class="section-header">👤 Customer Profile</div>', unsafe_allow_html=True)

tab_personal, tab_travel, tab_sales = st.tabs(
    ["🙋 Personal & Financial", "✈️ Travel Preferences", "📞 Sales Interaction"]
)

with tab_personal:
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_age = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)
        gender_input = st.selectbox("⚧ Gender", ["Male", "Female"])
    with col2:
        marital_status_input = st.selectbox(
            "💍 Marital Status", ["Single", "Married", "Divorced"]
        )
        job_type = st.selectbox(
            "💼 Occupation",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"],
        )
    with col3:
        job_title = st.selectbox(
            "🏷️ Designation", ["AVP", "VP", "Manager", "Senior Manager", "Executive"]
        )
        income_input = st.number_input(
            "💰 Monthly Income (₹)", min_value=0, value=25000, step=1000
        )

with tab_travel:
    col1, col2, col3 = st.columns(3)
    with col1:
        visitor_count = st.number_input(
            "🧑‍🤝‍🧑 Number of Persons Visiting", min_value=1, value=2
        )
        children_count = st.number_input(
            "🧒 Number of Children Visiting", min_value=0, value=1
        )
    with col2:
        trip_count = st.number_input("🗺️ Number of Trips (past)", min_value=0, value=2)
        property_star_rating = st.select_slider(
            "⭐ Preferred Property Star", options=[3, 4, 5], value=4
        )
    with col3:
        has_passport = st.radio("🛂 Has Passport?", ["No", "Yes"], horizontal=True)
        has_own_car = st.radio("🚗 Owns a Car?", ["No", "Yes"], horizontal=True)

with tab_sales:
    col1, col2, col3 = st.columns(3)
    with col1:
        contact_type = st.selectbox(
            "☎️ Type of Contact", ["Self Enquiry", "Company Invited"]
        )
        city_tier = st.select_slider("🏙️ City Tier", options=[1, 2, 3], value=1)
    with col2:
        pitch_duration = st.number_input(
            "⏱️ Duration of Pitch (minutes)", min_value=0, value=10
        )
        followup_count = st.number_input("🔁 Number of Followups", min_value=0, value=3)
    with col3:
        product_type = st.selectbox(
            "📦 Product Pitched",
            ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
        )
        satisfaction_score = st.select_slider(
            "😊 Pitch Satisfaction Score", options=[1, 2, 3, 4, 5], value=3
        )

st.markdown("<br>", unsafe_allow_html=True)


# --------------------------------------------------------------
# Assemble the record to score
# --------------------------------------------------------------

customer_record = pd.DataFrame(
    {
        "Age": [customer_age],
        "TypeofContact": [contact_type],
        "CityTier": [city_tier],
        "DurationOfPitch": [pitch_duration],
        "Occupation": [job_type],
        "Gender": [gender_input],
        "NumberOfPersonVisiting": [visitor_count],
        "NumberOfFollowups": [followup_count],
        "ProductPitched": [product_type],
        "PreferredPropertyStar": [property_star_rating],
        "MaritalStatus": [marital_status_input],
        "NumberOfTrips": [trip_count],
        "Passport": [1 if has_passport == "Yes" else 0],
        "PitchSatisfactionScore": [satisfaction_score],
        "OwnCar": [1 if has_own_car == "Yes" else 0],
        "NumberOfChildrenVisiting": [children_count],
        "Designation": [job_title],
        "MonthlyIncome": [income_input],
    }
)


# --------------------------------------------------------------
# Align columns with what the model was trained on
# --------------------------------------------------------------

def align_to_model_schema(record: pd.DataFrame, fitted_model) -> pd.DataFrame:
    required_columns = fitted_model.feature_names_in_

    for column_name in required_columns:
        if column_name not in record.columns:
            record[column_name] = 0

    return record[required_columns]


customer_record = align_to_model_schema(customer_record, predictor)


# --------------------------------------------------------------
# Run prediction and render results
# --------------------------------------------------------------

def render_prediction(fitted_model, record: pd.DataFrame) -> None:
    outcome = fitted_model.predict(record)[0]

    purchase_probability = None
    if hasattr(fitted_model, "predict_proba"):
        purchase_probability = fitted_model.predict_proba(record)[0][1]

    st.markdown('<div class="section-header">🔮 Prediction Result</div>', unsafe_allow_html=True)

    result_col, gauge_col = st.columns([1.3, 1])

    with result_col:
        if outcome == 1:
            st.markdown(
                f"""
                <div class="result-card-positive">
                    <div class="result-title">✅ Likely to Purchase</div>
                    <div class="result-subtitle">
                        This customer shows a strong profile match for buying
                        the tourism package. Consider prioritizing follow-up.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card-negative">
                    <div class="result-title">🚫 Unlikely to Purchase</div>
                    <div class="result-subtitle">
                        This customer's profile suggests a lower likelihood
                        of purchase. A tailored offer may help convert them.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if purchase_probability is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Purchase Probability", f"{purchase_probability:.1%}")
            m2.metric(
                "Confidence Level",
                "High" if abs(purchase_probability - 0.5) > 0.3 else "Moderate",
            )

    with gauge_col:
        if purchase_probability is not None:
            st.write("**Probability Gauge**")
            st.progress(float(purchase_probability))
            st.caption(
                f"{purchase_probability:.1%} chance of purchase, based on the "
                "customer's profile and interaction history."
            )

    with st.expander("📋 View Full Customer Profile Used for Prediction"):
        st.dataframe(record, use_container_width=True)


st.markdown('<div class="section-header">🚀 Ready to Predict</div>', unsafe_allow_html=True)
if st.button("Predict Package Purchase", type="primary"):
    render_prediction(predictor, customer_record)
else:
    st.info("👆 Fill in the customer details above, then click the button to see the prediction.")
