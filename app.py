import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('fracture_toughness_model.h5')
scaler = joblib.load('scaler.pkl')

# Streamlit app
def main():
    st.title("Fracture Toughness Prediction")

    st.markdown(
        "This app predicts the fracture toughness of sugarcane leaves and epoxy composites based on input factors."
    )

    # Input fields
    concentration = st.slider("Concentration (%wt.)", min_value=5, max_value=15, step=1, value=10)
    loading_rate = st.slider("Loading Rate (mm/min)", min_value=1, max_value=100, step=1, value=10)

    if st.button("Predict"):
        # Preprocess input
        input_data = np.array([[concentration, loading_rate]])
        input_scaled = scaler.transform(input_data)

        # Predict fracture toughness
        prediction = model.predict(input_scaled)

        # Display result
        st.success(f"Predicted Fracture Toughness: {prediction[0][0]:.4f} MPa m^(1/2)")

if __name__ == "__main__":
    main()
