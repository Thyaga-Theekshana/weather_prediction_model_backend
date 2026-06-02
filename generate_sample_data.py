import pandas as pd
import numpy as np
import datetime

# 1. Generate Timestamps minute-by-minute for 14 days (Total 20,160 rows)
start_date = datetime.datetime(2026, 5, 11, 0, 0, 0)
total_minutes = 14 * 24 * 60  
timestamps = [start_date + datetime.timedelta(minutes=i) for i in range(total_minutes)]

df = pd.DataFrame({'Timestamp': timestamps})
time_hours = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60.0

# Set seed for reproducible random data distribution
np.random.seed(42) 

# 2. Generate Synthetic Data matching Moratuwa's coastal-urban environment
# AHT10 - Temperature (26C - 33C) & Humidity (65% - 92%)
df['Temperature'] = 29.5 + 3.0 * np.sin((time_hours - 9) * np.pi / 12) + np.random.normal(0, 0.3, total_minutes)
df['Humidity'] = 78.5 - 11.5 * np.sin((time_hours - 9) * np.pi / 12) + np.random.normal(0, 0.6, total_minutes)

# BME280 - Atmospheric Pressure (1008 hPa - 1014 hPa)
df['Pressure'] = 1011.0 + 1.5 * np.sin((time_hours - 3) * np.pi / 12) + np.random.normal(0, 0.2, total_minutes)

# Rain Gauge - Rainfall amount (mm) - Intermittent rainy patches
rain_pattern = np.where(np.random.rand(total_minutes) > 0.98, np.random.exponential(1.5, total_minutes), 0.0)
df['Rainfall'] = np.round(rain_pattern, 2)

# BH1750 - Light Intensity (Lux) - Peaks during the day, drops to 0 at night
lux_pattern = 50000 * np.sin(time_hours * np.pi / 24)**4
df['Light_Intensity'] = np.where((time_hours > 6) & (time_hours < 18), lux_pattern + np.random.normal(0, 1000, total_minutes), 0.0)
df['Light_Intensity'] = df['Light_Intensity'].clip(lower=0)

# Anemometer & Wind Vane - Wind Speed and Direction
df['Wind_Speed'] = 3.5 + 2.0 * np.sin((time_hours - 10) * np.pi / 12) + np.random.normal(0, 0.5, total_minutes)
df['Wind_Speed'] = df['Wind_Speed'].clip(lower=0)
directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
df['Wind_Direction'] = np.random.choice(directions, total_minutes) # Pre-dominant SW/W coastal winds

# SGP41 - VOC & NOx (1-500 standard index range)
df['VOC_Index'] = 100 + 40 * np.sin((time_hours - 2) * np.pi / 12)**4 + np.random.randint(0, 30, total_minutes)
df['NOx_Index'] = 1 + 10 * np.sin((time_hours - 2) * np.pi / 12)**6 + np.random.randint(0, 5, total_minutes)

# PMS5003 - Particulate Matter (PM1.0, PM2.5, PM10) spikes during rush hours
df['PM2_5'] = 25 + 15 * np.sin((time_hours - 2) * np.pi / 12)**4 + np.random.exponential(3, total_minutes)
df['PM1_0'] = df['PM2_5'] * 0.65 + np.random.normal(0, 1, total_minutes)
df['PM10'] = df['PM2_5'] * 1.4 + np.random.normal(0, 2, total_minutes)

# MH-Z19C - CO2 (ppm) baseline around urban levels
df['CO2'] = 400 + 35 * np.sin((time_hours - 2) * np.pi / 12)**4 + np.random.normal(0, 4, total_minutes)

# MICS5524 - CO gas (ppm)
df['CO'] = 0.4 + 0.3 * np.sin((time_hours - 2) * np.pi / 12)**4 + np.random.normal(0, 0.05, total_minutes)

# Round all numerical columns to 2 decimal places
df = df.round(2)

# 3. Export dataset to a CSV file
file_name = 'moratuwa_sensors_20k_dataset.csv'
df.to_csv(file_name, index=False)

print(f"✅ Success! Created '{file_name}' with {len(df)} rows and {len(df.columns)} columns.")