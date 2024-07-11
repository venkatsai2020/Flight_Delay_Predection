import streamlit as st
import pandas as pd
import pickle as pk

#read live data
def read_data(path):
    df = pd.read_excel(path)
    return df

#function to load model
def load_ml_model(path):
    model = pk.load(open(path, "rb"))
    return model

#function to get airport list based on departing country
def feature_value_list(df, column):
    #get a list based on country
    value_list = df[column].astype('string').drop_duplicates()
    return value_list

#function to match delay
def get_matching_data_for_user_input(user_input, df):
    depart, arrival, dep_date, dep_time, airline, flight_number = user_input
    df["FROM"] = df["FROM"].astype('string')
    df["ARRIVAL_AIRPORT_NAME"] = df['ARRIVAL_AIRPORT_NAME'].astype('string')
    df['Scheduled_departures'] = df['Scheduled_departures'].astype('string')
    df['AIRLINE'] = df['AIRLINE'].astype('string')
    matching_rows = df[(df["FROM"] == depart) & (df['ARRIVAL_AIRPORT_NAME'] == arrival) & (df['Scheduled_departures'] == dep_time) & (df['AIRLINE'] == airline)]
    matching_row = matching_rows.drop_duplicates()
    return matching_row

def get_data_for_ui():
    df = read_data('E:/lyc.ca/term2/step/class_projects/liveData.xlsx')
    dep_list = feature_value_list(df, 'FROM')
    arrival_list = feature_value_list(df, 'ARRIVAL_AIRPORT_NAME')
    airlines = feature_value_list(df, 'AIRLINE')
    return dep_list, arrival_list, airlines

#function to predict delay
def predict(user_input):
    df = read_data('E:/lyc.ca/term2/step/class_projects/liveData.xlsx')
    model = load_ml_model('E:/lyc.ca/term2/step/class_projects/models/RandomForest.pk1')
    data  = get_matching_data_for_user_input(user_input, df)
    delay =  model.predict(data)
    return delay

def main():
    departure_airport_list, arrival_airport_list, airline_list = get_data_for_ui()

    st.set_page_config(
        page_title="Demo_Predict delay", 
        page_icon="")
    st.markdown("# Predict flight delay for your flight")
    st.write(
        """Enter your flight data and use the predict button to get predicted delay"""
    )

    col3, col4 = st.columns(2)
    with col3:
        depart = st.selectbox('departing airport', departure_airport_list)
        dep_date = st.date_input("Departing date")
    with col4:
        arrival = st.selectbox('arrival airport', arrival_airport_list)
        dep_time = st.time_input("Departing time")

    col1, col2= st.columns(2)
    with col1:
        airline = st.selectbox('airline', airline_list)
    with col2:
        flight_number = st.text_input('flight number')

    user_input = (depart, arrival, dep_date, dep_time, airline, flight_number)

    if st.button('Predcit Delay'):
        # value = predict(user_input)
        st.text("Predicted Delay:")

if __name__ == "__main__":
    main()
