import pandas as pd
import joblib

def preprocess_input(user_data):
    """
    Preprocess user input for prediction.
    1. Encode categorical fields using saved LabelEncoders.
    2. Align columns with training data.
    3. Scale numeric features.
    """
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([user_data])

    # Load encoders, scaler, and training columns
    encoders = joblib.load("models/label_encoders.pkl")
    scaler = joblib.load("models/scaler.pkl")
    training_columns = joblib.load("models/training_columns.pkl")

    # Apply label encoding
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Add missing columns
    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[training_columns]

    # Scale numeric features
    scaled_input = scaler.transform(input_df)

    return scaled_input
