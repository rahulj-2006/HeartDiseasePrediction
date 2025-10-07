import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

st.title("❤️ Heart Disease Prediction (Cached Model)")

# --- Load and cache dataset + model ---
@st.cache_data
def load_data():
    data = pd.read_csv("data/heart.csv")
    return data

@st.cache_resource
def train_model(data):
    # Detect target column
    target_col_candidates = [col for col in data.columns if col.lower() in ['target','output','heartdisease','disease']]
    if not target_col_candidates:
        st.error("❌ Could not find a target column!")
        st.stop()
    target_col = target_col_candidates[0]

    # Encode categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler, label_encoders, X

# Load data and train model once
data = load_data()
model, scaler, label_encoders, X_train_df = train_model(data)
st.success("✅ Model trained and cached!")

# --- User Input ---
st.subheader("Enter Your Details:")

def user_input():
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", 50, 250)
    chol = st.number_input("Cholesterol", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    sex_num = 1 if sex=="Male" else 0
    fbs_num = 1 if fbs=="Yes" else 0
    exang_num = 1 if exang=="Yes" else 0

    return {
        'Age': age,
        'Sex': sex_num,
        'ChestPainType': cp,
        'RestingBP': trestbps,
        'Cholesterol': chol,
        'FastingBS': fbs_num,
        'RestingECG': restecg,
        'MaxHR': thalach,
        'ExerciseAngina': exang_num,
        'Oldpeak': oldpeak,
        'ST_Slope': slope
    }

user_data = user_input()
input_df = pd.DataFrame([user_data])

# --- Preprocess input safely ---
for col, le in label_encoders.items():
    if col in input_df.columns:
        val = input_df.at[0, col]
        input_df[col] = le.transform([val])[0] if val in le.classes_ else 0

# Align columns
for col in X_train_df.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X_train_df.columns]

# Scale input
input_scaled = scaler.transform(input_df)

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    result = "💔 Heart Disease Detected" if prediction[0]==1 else "❤️ No Heart Disease"
    st.success(result)
