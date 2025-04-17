import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Step 1: Load the Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("accidents.csv")  # << your dataset file here
    return df

st.title("🚗 Accident Prediction App (Auto Data Import)")

# Load data
df = load_data()

st.subheader("Raw Data")
st.dataframe(df)

# --- Step 2: Data Cleaning ---

# Fill missing numeric columns with median
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical columns with mode
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop rows if 'Accident' column is missing (target column must not be null)
df = df.dropna(subset=["Accident"])

# Encoding categorical features
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

st.subheader("Cleaned Data")
st.dataframe(df)

# --- Step 3: Train the Model ---

X = df.drop("Accident", axis=1)
y = df["Accident"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"Model trained! ✅ Accuracy: {acc:.2f}")

# --- Step 4: Streamlit Input for New Predictions ---

st.subheader("Make a Prediction")

input_data = {}
for col in X.columns:
    if col in numeric_cols:
        input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].median()))
    else:
        options = list(df[col].unique())
        input_data[col] = st.selectbox(f"Select {col}", options)

if st.button("Predict Accident"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ Accident Predicted!")
    else:
        st.success("✅ No Accident Predicted!")
