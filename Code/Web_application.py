import streamlit as st
# import pandas as pd

# st.title("Flight Delay Prediction")
# df = pd.read_csv(r"C:\Users\kiran\Downloads\Airline_Delay_Cause.csv")

# enable_dropdown = st.radio('Enable Dropdown List', ['Yes', 'No'],index = 1)

# if enable_dropdown == 'Yes':
#     selected_columns = st.multiselect('select columns',df.columns)
#     if selected_columns:
#         st.write(df[selected_columns])
# else:
#     st.write('Dropdown List Disabled')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.wallpapersafari.com/1/47/XCW14A.jpg");
             background-size: cover;
             background-repeat: no-repeat;
             background-attachment: fixed;
             color: white; /* Ensure text is visible against the background */
             background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent black background */
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    

st.sidebar.title('Welcome!')
page = st.sidebar.radio('Go to',['Home', 'Dashboard','About'])

if page == 'Home':
    add_bg_from_url()
    st.title('Flight Delay Prediction')
    search_query = st.text_input('Enter Flight number')
elif page == 'About':
    st.write('Abour this Project:')
elif page == 'Dashboard':
    st.write('Welcome to the Flight Delay Prediction Dashboard!')

