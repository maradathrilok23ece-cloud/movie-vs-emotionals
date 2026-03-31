import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Movie Emotional Predictor", layout="centered")

st.title("🎬 Movie vs Emotional Attachment Predictor")

# -------- LOAD DATA --------
@st.cache_data
def load_data():
    data = pd.read_csv("Movie vs Emotional Attachment Survey 2024-2026.csv")
    return data

data = load_data()

st.subheader("📊 Dataset Preview")
st.write(data.head())

# -------- HANDLE MISSING VALUES --------
data = data.dropna()

# -------- ENCODE DATA --------
label_encoders = {}

for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# -------- SELECT TARGET --------
target_column = st.selectbox("🎯 Select Target Column", data.columns)

X = data.drop(target_column, axis=1)
y = data[target_column]

# -------- TRAIN MODEL --------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X.columns

model, feature_columns = train_model(X, y)

st.success("✅ Model Trained Successfully!")

# -------- USER INPUT --------
st.subheader("🧾 Enter Input Values")

input_data = {}

for col in feature_columns:
    input_data[col] = st.number_input(f"{col}", value=0)

input_df = pd.DataFrame([input_data])

# -------- PREDICTION --------
if st.button("🔮 Predict"):
    prediction = model.predict(input_df)[0]

    # Decode if target was categorical
    if target_column in label_encoders:
        prediction = label_encoders[target_column].inverse_transform([prediction])[0]

    st.success(f"🎯 Prediction: {prediction}")