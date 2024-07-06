import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Home",
    page_icon="",
)

st.title('Flight delay prediction')

st.write(
    """Welcome to the home page of our flight delay prediction project. 
    Navigate to the prediction page and enter your flight details and use the predict button to get our delay prediction. 
    """
)

