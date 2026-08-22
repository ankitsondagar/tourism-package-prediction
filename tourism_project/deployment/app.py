
"""
Streamlit front-end for the Tourism Package Prediction model.
Collects customer attributes through a simple form, builds a
single-row feature frame matching the trained model's schema,
and surfaces the predicted purchase outcome with its probability.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------------------
# Model loading
# --------------------------------------------------------------

def load_trained_model():
    model_file = Path(__file__).resolve().parent / "tourism_model.pkl"
    return joblib.load(model_file)


predictor = load_trained_model()


# --------------------------------------------------------------
# Page setup
# --------------------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    layout="wide",
)

st.title("Tourism Package Prediction")
st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase the tourism package."
)


# --------------------------------------------------------------
# Input form
# --------------------------------------------------------------

left_pane, right_pane = st.columns(2)

with left_pane:
    customer_age = st.number_input("Age", min_value=18, max_value=100, value=35)

    contact_type = st.selectbox(
        "Type of Contact", ["Self Enquiry", "Company Invited"]
    )

    city_tier = st.selectbox("City Tier", [1, 2, 3])

    pitch_duration = st.number_input("Duration of Pitch", min_value=0, value=10)

    job_type = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    )

    gender_input = st.selectbox("Gender", ["Male", "Female"])

    visitor_count = st.number_input(
        "Number of Persons Visiting", min_value=1, value=2
    )

    followup_count = st.number_input("Number of Followups", min_value=0, value=3)

    product_type = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
    )

with right_pane:
    property_star_rating = st.selectbox("Preferred Property Star", [3, 4, 5])

    marital_status_input = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced"]
    )

    trip_count = st.number_input("Number of Trips", min_value=0, value=2)

    has_passport = st.selectbox("Passport", [0, 1])

    satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

    has_own_car = st.selectbox("Own Car", [0, 1])

    children_count = st.number_input(
        "Number of Children Visiting", min_value=0, value=1
    )

    job_title = st.selectbox(
        "Designation", ["AVP", "VP", "Manager", "Senior Manager", "Executive"]
    )

    income_input = st.number_input("Monthly Income", min_value=0, value=25000)


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
        "Passport": [has_passport],
        "PitchSatisfactionScore": [satisfaction_score],
        "OwnCar": [has_own_car],
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

    st.subheader("Prediction")

    if outcome == 1:
        st.success("Customer is likely to purchase the tourism package.")
    else:
        st.info("Customer is unlikely to purchase the tourism package.")

    if hasattr(fitted_model, "predict_proba"):
        purchase_probability = fitted_model.predict_proba(record)[0][1]
        st.metric("Purchase Probability", f"{purchase_probability:.2%}")

    st.subheader("Customer Details")
    st.dataframe(record, use_container_width=True)


if st.button("Predict Package Purchase", type="primary"):
    render_prediction(predictor, customer_record)
