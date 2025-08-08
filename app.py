import streamlit as st
import numpy as np
import pandas as pd
import joblib # For loading the trained model and scaler
from datetime import date # To handle date inputs

# Suppress warnings for cleaner output in Streamlit
import warnings
warnings.simplefilter('ignore')

# --- 1. Model and Scaler Loading ---
# Ensure these files are in the same directory as your Streamlit app.py
# or provide the correct absolute/relative paths.
MODEL_PATH = 'rf_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURES_PATH = 'model_features.pkl'

try:
    loaded_rf_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_features = joblib.load(FEATURES_PATH)
    st.success("Model, scaler, and feature names loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model files not found. Please ensure '{MODEL_PATH}', '{SCALER_PATH}', and '{FEATURES_PATH}' are in the correct directory.")
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.error(f"An error occurred while loading model files: {e}")
    st.stop()

# Define categorical features and numerical columns for scaling, consistent with training
# These lists should match how they were defined and used in your training script.
CATEGORICAL_FEATURES = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
NUMERICAL_COLS_PRESENT = ['BHK', 'Size', 'Bathroom', 'month_posted', 'day_posted', 'day_of_week_posted', 'quarter_posted', 'Floor']

# --- 2. Preprocessing Function for New Data ---
def preprocess_new_data(input_data: dict, original_df_columns: list, scaler, categorical_features: list, numerical_cols_present: list) -> pd.DataFrame:
    """
    Preprocesses new input data to match the format expected by the trained model.
    This replicates the feature engineering steps from the training script.
    """
    # Create a DataFrame from the new input data
    new_df = pd.DataFrame([input_data])

    # --- Replicate Feature Engineering Steps ---

    # Process 'Posted On'
    new_df['month_posted'] = new_df['Posted On'].dt.month
    new_df['day_posted'] = new_df['Posted On'].dt.day
    new_df['day_of_week_posted'] = new_df['Posted On'].dt.day_of_week
    new_df['quarter_posted'] = new_df['Posted On'].dt.quarter
    new_df.drop('Posted On', axis=1, inplace=True)

    # Process 'Floor' column
    # Extract the first character for floor level
    new_df['Floor_Level'] = new_df['Floor'].astype(str).str[0]
    floor_mapping = {'L': -1, 'G': 0, 'U': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
                     '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12,
                     '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18,
                     '19': 19, '20': 20, '21': 21, '22': 22} # Extend as needed
    new_df['Floor'] = new_df['Floor_Level'].map(floor_mapping)
    new_df['Floor'].fillna(0, inplace=True) # Fill unmapped floors with 0
    new_df.drop('Floor_Level', axis=1, inplace=True)

    # Drop 'Area Locality' if it exists in the input (it was dropped during training)
    if 'Area Locality' in new_df.columns:
        new_df.drop('Area Locality', axis=1, inplace=True)

    # Apply one-hot encoding
    for feature in categorical_features:
        if feature in new_df.columns:
            new_df = pd.get_dummies(new_df, columns=[feature], drop_first=True)

    # Align columns with the training data (CRITICAL for consistent input to model)
    # Add missing columns with 0, and drop extra columns that weren't in training data
    missing_cols = set(original_df_columns) - set(new_df.columns)
    for c in missing_cols:
        new_df[c] = 0
    # Ensure the order of columns is exactly the same as during training
    new_df = new_df[original_df_columns]

    # Scale numerical features
    new_df[numerical_cols_present] = scaler.transform(new_df[numerical_cols_present])

    return new_df

# --- 3. Streamlit Application Layout ---
st.set_page_config(
    page_title="House Rent Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🏠 House Rent Prediction")
st.markdown("Enter the details of a house to get an estimated monthly rent.")

# Input widgets for house features
st.subheader("Property Details")

col1, col2 = st.columns(2)
with col1:
    bhk = st.number_input("BHK (Bedrooms, Hall, Kitchen)", min_value=1, max_value=10, value=2, step=1)
    bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    area_type = st.selectbox("Area Type", ['Super Area', 'Carpet Area', 'Built Area'])
    furnishing_status = st.selectbox("Furnishing Status", ['Unfurnished', 'Semi-Furnished', 'Furnished'])

with col2:
    size = st.number_input("Size (in sq ft)", min_value=100, max_value=50000, value=1000, step=50)
    floor_input = st.text_input("Floor (e.g., 'Ground out of 2', '5 out of 10')", value="1 out of 3")
    city = st.selectbox("City", ['Kolkata', 'Mumbai', 'Bangalore', 'Delhi', 'Hyderabad', 'Chennai'])
    tenant_preferred = st.selectbox("Tenant Preferred", ['Bachelors/Family', 'Bachelors', 'Family'])

posted_on = st.date_input("Posted On Date", value=date.today())
point_of_contact = st.selectbox("Point of Contact", ['Contact Owner', 'Contact Agent'])

# Button to trigger prediction
if st.button("Predict Rent"):
    if loaded_rf_model and loaded_scaler and loaded_features:
        input_data = {
            'BHK': bhk,
            'Size': size,
            'Bathroom': bathroom,
            'Floor': floor_input,
            'Area Type': area_type,
            'City': city,
            'Furnishing Status': furnishing_status,
            'Tenant Preferred': tenant_preferred,
            'Point of Contact': point_of_contact,
            'Posted On': pd.to_datetime(posted_on), # Convert date object to datetime
            'Area Locality': 'N/A' # Placeholder, as it's dropped
        }

        try:
            # Preprocess the input data
            processed_input = preprocess_new_data(
                input_data,
                loaded_features, # Use the feature names saved during training
                loaded_scaler,
                CATEGORICAL_FEATURES,
                NUMERICAL_COLS_PRESENT
            )

            # Make prediction
            log_predicted_rent = loaded_rf_model.predict(processed_input)[0]
            predicted_rent = np.expm1(log_predicted_rent) # Inverse transform log rent

            st.subheader("Prediction Result:")
            st.success(f"Estimated Monthly Rent: **₹{predicted_rent:,.2f}**") # Using INR symbol

            # --- Price Classification ---
            st.markdown("---")
            st.subheader("Price Comparison")
            st.markdown("Enter a listed price to see if the property is underpriced, overpriced, or fairly priced based on our prediction.")

            listed_price = st.number_input("Enter Listed Price (₹)", min_value=0, value=int(predicted_rent * 1.05), step=100)

            FAIR_PRICE_TOLERANCE = 0.10 # 10% tolerance

            lower_bound = predicted_rent * (1 - FAIR_PRICE_TOLERANCE)
            upper_bound = predicted_rent * (1 + FAIR_PRICE_TOLERANCE)

            st.info(f"A fair price for this property would typically be between **₹{lower_bound:,.2f}** and **₹{upper_bound:,.2f}**.")

            if listed_price < lower_bound:
                st.warning("This property appears to be **Underpriced**! Great deal!")
            elif listed_price > upper_bound:
                st.error("This property appears to be **Overpriced**! Consider negotiating.")
            else:
                st.success("This property appears to be **Fairly Priced**.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check your input values and ensure the model files are correct.")
    else:
        st.warning("Model not loaded. Please check the file paths and restart the app.")

# --- How to Run Instructions ---
st.markdown(
    """
    ---
    ### How to Run This Application:
    1.  **Save** the code above as a Python file (e.g., `rent_app.py`).
    2.  **Ensure** you have run your original training script to generate the model files:
        `rf_model.pkl`, `scaler.pkl`, and `model_features.pkl`.
    3.  **Place** these three `.pkl` files in the **same directory** as `rent_app.py`.
    4.  **Install** the necessary libraries:
        ```bash
        pip install streamlit numpy pandas scikit-learn tensorflow # tensorflow is not strictly needed for prediction if model is joblib saved
        ```
        *(Note: `tensorflow` is not strictly required if your model was saved with `joblib` and doesn't rely on `tf.keras.models.load_model` for inference, but it's good to have if your original model was a Keras model.)*
    5.  **Run** the app from your terminal:
        ```bash
        streamlit run rent_app.py
        ```
    6.  Your browser will automatically open to the Streamlit app!
    """
)
