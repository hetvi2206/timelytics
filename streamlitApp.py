# Import required libraries
import streamlit as st
import pickle
import numpy as np
import gdown  # For downloading from Google Drive
import os
from PIL import Image

# Set the page configuration
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

# Display the title and captions
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times. By combining the strengths of these three algorithms, Timelytics provides a robust and reliable prediction of OTD times, helping businesses to optimize their supply chain operations."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them, reducing lead times and improving delivery times. The model utilizes historical data on order processing times, production lead times, shipping times, and other relevant variables to generate accurate forecasts of OTD times. These forecasts can be used to optimize inventory management, improve customer service, and increase overall efficiency in the supply chain."
)

# Google Drive file ID for voting_model.pkl
file_id = "1reTM7fTqgScx6X0ddwHOvwACAEy_3Asd"  # Replace with your actual file ID
model_path = "voting_model.pkl"

# Function to download the model if it doesn't exist
def download_model():
    if not os.path.exists(model_path):
        st.info("Downloading model from Google Drive...")
        url = f"https://drive.google.com/file/d/1reTM7fTqgScx6X0ddwHOvwACAEy_3Asd/view?usp=sharing"
        gdown.download(url, model_path, quiet=False)

# Download the model
download_model()

# Load the model
try:
    with open(model_path, "rb") as f:
        voting_model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Caching the model
@st.cache_resource
def get_model():
    return voting_model

# Function for wait time prediction
def waitime_predictor(
    purchase_dow, purchase_month, year, product_size_cm3, product_weight_g, 
    geolocation_state_customer, geolocation_state_seller, distance
):
    model = get_model()
    prediction = model.predict(
        np.array([
            [
                purchase_dow, purchase_month, year, product_size_cm3, product_weight_g, 
                geolocation_state_customer, geolocation_state_seller, distance
            ]
        ])
    )
    return round(prediction[0])

# Sidebar input parameters
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm^3", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)
    submit = st.button(label="Predict Wait Time!")

# Output container for predictions
with st.container():
    st.header("Output: Wait Time in Days")
    if submit:
        prediction = waitime_predictor(
            purchase_dow, purchase_month, year, product_size_cm3, product_weight_g, 
            geolocation_state_customer, geolocation_state_seller, distance
        )
        with st.spinner(text="This may take a moment..."):
            st.write(prediction)

# Sample dataset display
import pandas as pd
data = {
    "Purchased Day of the Week": ["0", "3", "1"],
    "Purchased Month": ["6", "3", "1"],
    "Purchased Year": ["2018", "2017", "2018"],
    "Product Size in cm^3": ["37206.0", "63714", "54816"],
    "Product Weight in grams": ["16250.0", "7249", "9600"],
    "Geolocation State Customer": ["25", "25", "25"],
    "Geolocation State Seller": ["20", "7", "20"],
    "Distance": ["247.94", "250.35", "4.915"],
}
df = pd.DataFrame(data)
st.header("Sample Dataset")
st.write(df)
