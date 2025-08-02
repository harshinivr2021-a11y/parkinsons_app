import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
df = pd.read_csv("parkinsons.csv")
X = df.drop(['status', 'name'], axis=1)
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

st.title("🧠 Parkinson's Disease Prediction")

st.write("Enter the following voice-related metrics:")

fo = st.number_input("MDVP:Fo(Hz)")
fhi = st.number_input("MDVP:Fhi(Hz)")
flo = st.number_input("MDVP:Flo(Hz)")
jitter = st.number_input("MDVP:Jitter(%)")
shimmer = st.number_input("MDVP:Shimmer")
hnr = st.number_input("HNR")
spread1 = st.number_input("spread1")
d2 = st.number_input("D2")

input_data = [fo, fhi, flo, jitter, shimmer, hnr, spread1, d2]

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    st.success("🟢 No Parkinson's Detected" if prediction[0] == 0 else "🔴 Parkinson's Detected")
