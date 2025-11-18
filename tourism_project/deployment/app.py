import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="affanthinks/Tourism-Package-Prediction", filename="best_tourism_pred_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("tourism Prediction App")
st.write("The tourism Prediction App is an internal tool for tourism staff that predicts whether customers are purchasing the product based on their details and pitch.")
st.write("Kindly enter the customer details to check whether they are likely to purchase.")

# Collect user input
Age = st.number_input("Age (Age of the customer)", min_value=18, max_value=120, value=30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Fe Male"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1, max_value=20, value=2
)

PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])


MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Divorced", "Married", "Unmarried"]
)

NumberOfTrips = st.number_input("Number of Trips Annually", min_value=1, max_value=22, value=1)


Passport = st.selectbox(
    "Passport",
    ["Yes", "No"]
)

OwnCar = st.selectbox(
    "Own Car",
    ["Yes", "No"]
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting (below 5)",
    min_value=0, max_value=10, value=0
)

Designation = st.selectbox(
    "Designation",
    ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000.0, value=50000.0
)

# NEW REQUIRED FIELDS
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)


ProductPitched = st.selectbox(
    "Product Pitched",
    ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"]
)

NumberOfFollowups = st.number_input(
    "Number Of Follow-ups",
    min_value=0, max_value=50, value=1
)

DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=127, value=10)


# Create dataframe for model
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    # REQUIRED new fields
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])


# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
