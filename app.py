import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Step 1: Load the Dataset ---
@st.cache
def load_data():
    df = pd.read_csv("dataset_traffic_accident_prediction1.csv")  # << your dataset file here
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

# Define options for each column based on the dataset
weather_options = ['Rainy', 'Clear', 'Foggy', 'Stormy', 'Snowy']
road_type_options = ['City Road', 'Rural Road', 'Highway', 'Mountain Road']
time_of_day_options = ['Morning', 'Afternoon', 'Evening', 'Night']
traffic_density_options = [0, 1, 2]
road_condition_options = ['Dry', 'Wet', 'Icy', 'Under Construction']
vehicle_type_options = ['Car', 'Truck', 'Bus', 'Motorcycle']
road_light_condition_options = ['Daylight', 'Artificial Light', 'No Light']

# Take input from the user for prediction
input_data = {
    "Weather": st.selectbox("Select Weather", weather_options),
    "Road_Type": st.selectbox("Select Road Type", road_type_options),
    "Time_of_Day": st.selectbox("Select Time of Day", time_of_day_options),
    "Traffic_Density": st.selectbox("Select Traffic Density", traffic_density_options),
    "Speed_Limit": st.number_input("Enter Speed Limit", value=int(df['Speed_Limit'].median())),
    "Number_of_Vehicles": st.number_input("Enter Number of Vehicles", value=int(df['Number_of_Vehicles'].median())),
    "Driver_Alcohol": st.number_input("Enter Driver Alcohol (0 = No, 1 = Yes)", value=int(df['Driver_Alcohol'].median())),
    "Accident_Severity": st.selectbox("Select Accident Severity", ['Low', 'Moderate', 'High']),
    "Road_Condition": st.selectbox("Select Road Condition", road_condition_options),
    "Vehicle_Type": st.selectbox("Select Vehicle Type", vehicle_type_options),
    "Driver_Age": st.number_input("Enter Driver Age", value=int(df['Driver_Age'].median())),
    "Driver_Experience": st.number_input("Enter Driver Experience", value=int(df['Driver_Experience'].median())),
    "Road_Light_Condition": st.selectbox("Select Road Light Condition", road_light_condition_options),
}

if st.button("Predict Accident"):
    # Encoding the categorical inputs before prediction
    input_df = pd.DataFrame([input_data])
    
    # Encoding categorical features
    input_df['Weather'] = le.transform(input_df['Weather'])
    input_df['Road_Type'] = le.transform(input_df['Road_Type'])
    input_df['Time_of_Day'] = le.transform(input_df['Time_of_Day'])
    input_df['Road_Condition'] = le.transform(input_df['Road_Condition'])
    input_df['Vehicle_Type'] = le.transform(input_df['Vehicle_Type'])
    input_df['Accident_Severity'] = le.transform(input_df['Accident_Severity'])
    input_df['Road_Light_Condition'] = le.transform(input_df['Road_Light_Condition'])

    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ Accident Predicted!")
    else:
        st.success("✅ No Accident Predicted!")
