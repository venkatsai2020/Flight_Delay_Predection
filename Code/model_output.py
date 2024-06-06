import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load the model
model_path = r"C:\Users\kiran\OneDrive\Desktop\Files\Programing\Ml_Project_Venkat\Flight_Delay_Predection\PickledModels\FlightDelayPredecation_LinearRegression05062024113252.pk1"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the dataset to get default values
data_path = r"C:\Users\kiran\Downloads\Airline_Delay_Cause (1).csv"
data = pd.read_csv(data_path)

# Define a function to get default values for imputation
def get_default_values():
    defaults = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            defaults[column] = data[column].mode()[0]  # Using mode as default value (mode imputation)
        else:
            defaults[column] = data[column].mean()  # Using mean as default value (mean imputation)
    return defaults

default_values = get_default_values()

# Streamlit app
st.title('Flight Delay Prediction App')

# Create input fields for user-provided parameters
year = st.selectbox('Year', data['year'].unique())
month = st.selectbox('Month', data['month'].unique())
carrier = st.selectbox('Carrier', data['carrier'].unique())
carrier_name = st.selectbox('Carrier Name', data['carrier_name'].unique())
airport = st.selectbox('Airport', data['airport'].unique())
airport_name = st.selectbox('Airport Name', data['airport_name'].unique())
arr_flights = st.number_input('Arrival Flights', value=int(default_values['arr_flights']))
# carrier_ct = st.number_input('Carrier Count', value=int(default_values['carrier_ct']))
# weather_ct = st.number_input('Weather Count', value=int(default_values['weather_ct']))
# nas_ct = st.number_input('NAS Count', value=int(default_values['nas_ct']))
# security_ct = st.number_input('Security Count', value=int(default_values['security_ct']))
# late_aircraft_ct = st.number_input('Late Aircraft Count', value=int(default_values['late_aircraft_ct']))
# arr_cancelled = st.number_input('Arrival Cancelled', value=int(default_values['arr_cancelled']))
# arr_diverted = st.number_input('Arrival Diverted', value=int(default_values['arr_diverted']))
# carrier_delay = st.number_input('Carrier Delay', value=int(default_values['carrier_delay']))
# weather_delay = st.number_input('Weather Delay', value=int(default_values['weather_delay']))
# nas_delay = st.number_input('NAS Delay', value=int(default_values['nas_delay']))
# security_delay = st.number_input('Security Delay', value=int(default_values['security_delay']))
# late_aircraft_delay = st.number_input('Late Aircraft Delay', value=int(default_values['late_aircraft_delay']))

if st.button('Predict'):
    # Create a dictionary to hold all feature values
    input_features = {
        'year': year,
        'month': month,
        'arr_flights': arr_flights,
        'carrier': carrier,
        'carrier_name': carrier_name,
        'airport': airport,
        'airport_name': airport_name,
        # 'carrier_ct': carrier_ct,
        # 'weather_ct': weather_ct,
        # 'nas_ct': nas_ct,
        # 'security_ct': security_ct,
        # 'late_aircraft_ct': late_aircraft_ct,
        # 'arr_cancelled': arr_cancelled,
        # 'arr_diverted': arr_diverted,
        # 'carrier_delay': carrier_delay,
        # 'weather_delay': weather_delay,
        # 'nas_delay': nas_delay,
        # 'security_delay': security_delay,
        # 'late_aircraft_delay': late_aircraft_delay
    }

    # Convert input features to a DataFrame
    input_df = pd.DataFrame([input_features])

    # Ensure that all required features are included in the DataFrame with the correct names
    for column in model.feature_names_in_:
        if column not in input_df.columns:
            input_df[column] = default_values[column]

    # Make prediction
    prediction = model.predict(input_df)

    # Display the result
    st.write(f'The predicted flight delay is: {prediction[0]} minutes')

# Run the app using the following command in the terminal:
# streamlit run app.py