import streamlit as st
from predictor import FireRiskPredictor

# Initialize and train the predictor only once
@st.cache_resource
def load_predictor():
    predictor = FireRiskPredictor('forestfires.csv')
    predictor.preprocess_data()
    predictor.train_model()
    return predictor

fire_predictor = load_predictor()

st.title("Forest Fire Risk Prediction")
st.write("Enter today's weather and forest conditions to predict fire risk.")

with st.form("fire_form"):
    temp = st.number_input("Temperature (°C)", min_value=-30.0, max_value=50.0, value=20.0, step=0.1)
    RH = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
    wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    FFMC = st.number_input("Fine Fuel Moisture Code (FFMC)", min_value=0.0, max_value=150.0, value=85.0, step=0.1)
    DMC = st.number_input("Duff Moisture Code (DMC)", min_value=0.0, max_value=300.0, value=50.0, step=0.1)
    DC = st.number_input("Drought Code (DC)", min_value=0.0, max_value=1000.0, value=100.0, step=0.1)
    ISI = st.number_input("Initial Spread Index (ISI)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    submitted = st.form_submit_button("Predict Fire Risk")

if submitted:
    input_data = {
        'temp': temp,
        'RH': RH,
        'wind': wind,
        'rain': rain,
        'FFMC': FFMC,
        'DMC': DMC,
        'DC': DC,
        'ISI': ISI
    }
    try:
        prediction = fire_predictor.predict_fire_risk(input_data)
        st.subheader("Prediction Result")
        st.write(f"**Risk Level:** {prediction['risk_level']}")
        st.write(f"**Fire Probability:** {prediction['fire_probability']*100:.2f}%")
    except Exception as e:
        st.error(f"Error: {e}")
