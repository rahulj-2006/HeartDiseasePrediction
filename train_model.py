import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv('data/heart.csv')

# Detect target column automatically
target_col_candidates = [col for col in data.columns if col.lower() in ['target', 'output', 'heartdisease', 'disease']]
if not target_col_candidates:
    raise ValueError("❌ Could not find a target column.")
target_col = target_col_candidates[0]
print(f"🎯 Using '{target_col}' as target column.")

# Encode categorical columns
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
if cat_cols:
    print(f"🔠 Encoding categorical columns: {cat_cols}")
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Split features and target
X = data.drop(target_col, axis=1)
y = data[target_col]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# Save everything
os.makedirs("models", exist_ok=True)
joblib.dump(model, 'models/heart_disease_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'models/training_columns.pkl')

print("💾 Model, scaler, encoders, and training columns saved in /models/")
