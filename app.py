import streamlit as st
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import json
from PIL import Image

st.title('Deforestation on Kalimantan Island')
def get_df(path):
    # Load data from JSON file
    with open(path) as f:
        data = json.load(f)

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Value'])
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')  # Convert to datetime with only year
    df.set_index('Year', inplace=True)
    # Title of the webpage
    
    # Fit ARIMA model
    model = auto_arima(df['Value'], start_p=1, start_q=1, max_p=3, max_q=3,
                    d=2, seasonal=False, trace=True,
                    error_action='ignore', suppress_warnings=True, stepwise=True)
    # Predict for the next 5 years
    n_periods = 5
    forecast_years = pd.date_range(start=df.index.max(), periods=n_periods + 1, freq='YS')[1:]
    forecast, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    # Optional: Add future predictions to the chart
    future_df = pd.DataFrame(forecast, index=forecast_years, columns=['Value'])
    full_df = pd.concat([df, future_df])
    return full_df

df_loss = get_df("data_timeseries_loss.json")
df_gain = get_df("data_timeseries_gain.json")

selected_chart = st.selectbox('Select a chart:', list(["loss","gain"]))  # Update range according to your files

# Visualization
if selected_chart == "loss":
    st.write("## Loss Visualization with ARIMA Predictions")
    st.line_chart(df_loss)
else:
    st.write("## Gain Visualization with ARIMA Predictions")
    st.line_chart(df_gain)

# Dropdown for image selection
st.write("## Yearly Image Viewer")
selected_year = st.selectbox('Select a Year:', list(range(2001, 2023 + 1)))  # Update range according to your files

# Display the selected year image
image_folder = 'pictures'
image_path = f'{image_folder}/{selected_year}.jpg'

# Load and display the image
try:
    image = Image.open(image_path)
    st.image(image, caption=f'Image for Year {selected_year}')
except FileNotFoundError:
    st.error(f"No image found for year {selected_year}.")



st.image("https://www.globalforestwatch.org/_next/static/images/indonesia-primary-forest-loss-2023-8449f26485c9e4e8a2667a6ba5320124.png")
import streamlit.components.v1 as components

# embed streamlit docs in a streamlit app
components.iframe("https://www.globalforestwatch.org/embed/widget/treeLoss/country/IDN", height=500)
