import streamlit as st
import pandas as pd

st.title("Flight Delay Prediction")
df = pd.read_csv(r"C:\Users\kiran\Downloads\Airline_Delay_Cause.csv")

enable_dropdown = st.radio('Enable Dropdown List', ['Yes', 'No'],index = 1)

if enable_dropdown == 'Yes':
    selected_columns = st.multiselect('select columns',df.columns)
    if selected_columns:
        st.write(df[selected_columns])
else:
    st.write('Dropdown List Disabled')

