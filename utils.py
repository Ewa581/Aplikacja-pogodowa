import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate realistic sample weather data"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=365*3),
        end=datetime.now(),
        freq='D'
    )
    
    np.random.seed(42)
    
    # Temperature with seasonal pattern
    base_temp = 10 + 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    temperature = base_temp + np.random.normal(0, 3, len(dates))
    
    # Rainfall with seasonal probability
    rain_prob = 0.3 + 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    rain = np.where(
        np.random.random(len(dates)) < rain_prob,
        np.random.gamma(shape=1, scale=2, size=len(dates)),
        0
    )
    
    # Wind speed
    wind_speed = np.random.weibull(2, size=len(dates)) * 10
    
    return pd.DataFrame({
        "date": dates,
        "temperature_C": np.round(temperature, 1),
        "rain_mm": np.round(rain, 1),
        "wind_speed_kmh": np.round(wind_speed, 1),
        "location": "Sample City"
    })

def load_data(uploaded_file):
    """Load data from uploaded file or generate sample"""
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            # Ensure required columns exist
            if 'date' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'date'})
            return df
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    else:
        return generate_sample_data()