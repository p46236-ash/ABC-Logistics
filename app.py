import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Input fields for each feature
delivery_distance = st.number_input('Delivery Distance (km)', min_value=0.0, max_value=100.0, value=25.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5, e.g., 1=Clear, 5=Severe)', 1, 5, 2)
delivery_slot = st.slider('Delivery Slot (1-3, e.g., 1=Morning, 2=Afternoon, 3=Evening)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 30, 5)
num_stops = st.slider('Number of Stops', 0, 15, 3)
vehicle_age = st.slider('Vehicle Age (years)', 0, 20, 5)
road_condition_score = st.slider('Road Condition Score (1-5, e.g., 1=Poor, 5=Excellent)', 1, 5, 3)
package_weight = st.number_input('Package Weight (kg)', min_value=0.1, max_value=50.0, value=10.0)
fuel_efficiency = st.number_input('Fuel Efficiency (km/L)', min_value=5.0, max_value=30.0, value=15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 120, 30)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.error('Prediction: **DELIVERY LIKELY DELAYED**')
    else:
        st.success('Prediction: **DELIVERY LIKELY ON TIME**')
    
    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")

st.markdown("""
## How to run this app:
1. Save this code as `streamlit_app.py` (which I've done for you).
2. Open your terminal or command prompt.
3. Navigate to the directory where `streamlit_app.py` is saved.
4. Run the command: `streamlit run streamlit_app.py`
""")
