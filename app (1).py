
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained XGBoost model
finalmodel = joblib.load('xgboost_model.joblib')

st.title('Customer Churn Prediction App')
st.write('Enter the customer details below to predict churn.')

# Define the ordered list of features expected by the model
# This order is derived from the 'x' DataFrame after preprocessing
feature_columns = [
    'Tenure',
    'CityTier',
    'WarehouseToHome',
    'HourSpendOnApp',
    'NumberOfDeviceRegistered',
    'SatisfactionScore',
    'NumberOfAddress',
    'Complain',
    'OrderAmountHikeFromlastYear',
    'CouponUsed',
    'OrderCount',
    'DaySinceLastOrder',
    'CashbackAmount',
    'Gender_Female',
    'Gender_Male',
    'MaritalStatus_Divorced',
    'MaritalStatus_Married',
    'MaritalStatus_Single'
]

# Create input widgets for numerical features
input_data = {}

# Numerical inputs with reasonable ranges and defaults
input_data['Tenure'] = st.number_input('Tenure (months)', min_value=0.0, max_value=100.0, value=10.0, step=1.0)
input_data['CityTier'] = st.number_input('City Tier', min_value=1.0, max_value=3.0, value=1.0, step=1.0)
input_data['WarehouseToHome'] = st.number_input('Warehouse To Home (miles)', min_value=5.0, max_value=150.0, value=15.0, step=1.0)
input_data['HourSpendOnApp'] = st.number_input('Hours Spend On App', min_value=0.0, max_value=5.0, value=3.0, step=0.5)
input_data['NumberOfDeviceRegistered'] = st.number_input('Number Of Devices Registered', min_value=1.0, max_value=6.0, value=3.0, step=1.0)
input_data['SatisfactionScore'] = st.number_input('Satisfaction Score (1-5)', min_value=1.0, max_value=5.0, value=3.0, step=1.0)
input_data['NumberOfAddress'] = st.number_input('Number Of Addresses', min_value=1.0, max_value=25.0, value=4.0, step=1.0)

# Binary input for Complain
complain_option = st.radio('Has the customer complained in last 3 months?', ('No', 'Yes'))
input_data['Complain'] = 1.0 if complain_option == 'Yes' else 0.0

input_data['OrderAmountHikeFromlastYear'] = st.number_input('Order Amount Hike From Last Year (%)', min_value=10.0, max_value=30.0, value=15.0, step=1.0)
input_data['CouponUsed'] = st.number_input('Coupons Used', min_value=0.0, max_value=20.0, value=1.0, step=1.0)
input_data['OrderCount'] = st.number_input('Order Count', min_value=1.0, max_value=20.0, value=3.0, step=1.0)
input_data['DaySinceLastOrder'] = st.number_input('Days Since Last Order', min_value=0.0, max_value=50.0, value=5.0, step=1.0)
input_data['CashbackAmount'] = st.number_input('Cashback Amount', min_value=0.0, max_value=350.0, value=150.0, step=1.0)

# Categorical inputs for Gender and MaritalStatus with one-hot encoding
gender_options = ['Male', 'Female']
selected_gender = st.radio('Gender', gender_options)
input_data['Gender_Female'] = 1.0 if selected_gender == 'Female' else 0.0
input_data['Gender_Male'] = 1.0 if selected_gender == 'Male' else 0.0

marital_status_options = ['Single', 'Married', 'Divorced']
selected_marital_status = st.radio('Marital Status', marital_status_options)
input_data['MaritalStatus_Divorced'] = 1.0 if selected_marital_status == 'Divorced' else 0.0
input_data['MaritalStatus_Married'] = 1.0 if selected_marital_status == 'Married' else 0.0
input_data['MaritalStatus_Single'] = 1.0 if selected_marital_status == 'Single' else 0.0


if st.button('Predict Churn'):
    # Create a DataFrame from the input data, ensuring correct column order
    # Initialize with zeros for all OHE columns first to avoid key errors if a category isn't chosen
    processed_input = {col: 0.0 for col in feature_columns}
    for key, value in input_data.items():
        if key in processed_input:
            processed_input[key] = value

    # Ensure the DataFrame has the correct columns in the right order
    input_df = pd.DataFrame([processed_input], columns=feature_columns)

    # Convert DataFrame to a NumPy array for prediction
    input_array = input_df.values

    # Make prediction
    prediction = finalmodel.predict(input_array)

    if prediction[0] == 1:
        st.error('The customer is likely to Churn!')
    else:
        st.success('The customer is unlikely to Churn.')
