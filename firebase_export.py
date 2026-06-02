"""
Firebase Data Export Script
Exports weather data from Firebase Realtime Database to CSV for ML training
"""

import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import json
from datetime import datetime
import os

import config

# Initialize Firebase Admin SDK
def init_firebase(credentials_path=None, database_url=None):
    """Initialize Firebase with service account credentials."""
    credentials_path = credentials_path or config.FIREBASE_CREDENTIALS
    database_url = database_url or config.FIREBASE_DATABASE_URL

    if not database_url:
        raise ValueError('FIREBASE_DATABASE_URL is required. Set it in .env or your environment.')

    if credentials_path and os.path.exists(credentials_path):
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url
        })
    else:
        firebase_admin.initialize_app(options={
            'databaseURL': database_url
        })

    print("✓ Firebase initialized")

def fetch_all_data():
    """Fetch all weather data from Firebase"""
    ref = db.reference('/weather_data')
    data = ref.get()
    
    if not data:
        print("No data found in Firebase")
        return None
    
    print(f"✓ Fetched {len(data)} records from Firebase")
    return data

def flatten_data(firebase_data):
    """Convert nested Firebase structure to flat DataFrame"""
    records = []
    
    for timestamp, reading in firebase_data.items():
        record = {
            'timestamp': int(timestamp),
            'datetime': datetime.fromtimestamp(int(timestamp)),
            'device_id': reading.get('device_id', 'unknown')
        }
        
        # Extract weather data
        if 'weather' in reading:
            weather = reading['weather']
            record.update({
                'temperature': weather.get('temperature'),
                'humidity': weather.get('humidity'),
                'pressure': weather.get('pressure'),
                'rainfall': weather.get('rainfall'),
                'light_intensity': weather.get('light_intensity'),
                'wind_speed': weather.get('wind_speed'),
                'wind_direction': weather.get('wind_direction')
            })
        
        # Extract air quality data
        if 'air_quality' in reading:
            air = reading['air_quality']
            record.update({
                'voc': air.get('voc'),
                'nox': air.get('nox'),
                'pm1_0': air.get('pm1_0'),
                'pm2_5': air.get('pm2_5'),
                'pm10': air.get('pm10'),
                'co2': air.get('co2'),
                'co': air.get('co')
            })
        
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✓ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df

def clean_data(df):
    """Clean and validate sensor data"""
    print("\nData Cleaning:")
    print(f"  Initial rows: {len(df)}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    print(f"  After removing duplicates: {len(df)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n  Missing values:")
        print(missing[missing > 0])
    
    # Remove rows with too many missing values (>50%)
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    print(f"  After removing incomplete rows: {len(df)}")
    
    # Forward fill remaining missing values (sensor readings)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    
    # Validate sensor ranges (remove obvious outliers)
    df = df[
        (df['temperature'] >= -50) & (df['temperature'] <= 60) &
        (df['humidity'] >= 0) & (df['humidity'] <= 100) &
        (df['pressure'] >= 900) & (df['pressure'] <= 1100)
    ]
    print(f"  After outlier removal: {len(df)}")
    
    return df

def add_time_features(df):
    """Add temporal features for ML"""
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print(f"✓ Added time features")
    return df

def export_to_csv(df, output_dir='data'):
    """Export processed data to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{output_dir}/weather_data_{timestamp}.csv'
    
    df.to_csv(filename, index=False)
    print(f"\n✓ Exported data to: {filename}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    return filename

def main():
    """Main execution"""
    print("=== Firebase Data Export ===\n")

    credentials_path = config.FIREBASE_CREDENTIALS
    database_url = config.FIREBASE_DATABASE_URL

    if not credentials_path or not database_url:
        print("ERROR: Missing Firebase configuration.")
        print("Copy .env.example to .env and set FIREBASE_DATABASE_URL and FIREBASE_CREDENTIALS.")
        return

    if not os.path.exists(credentials_path):
        print(f"ERROR: Firebase credentials not found at {credentials_path}")
        return

    # Initialize Firebase
    init_firebase(credentials_path, database_url)
    
    # Fetch data
    data = fetch_all_data()
    if not data:
        return
    
    # Convert to DataFrame
    df = flatten_data(data)
    
    # Clean data
    df = clean_data(df)
    
    # Add time features
    df = add_time_features(df)
    
    # Export to CSV
    export_to_csv(df)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(df[['temperature', 'humidity', 'pressure', 'wind_speed']].describe())

if __name__ == '__main__':
    import numpy as np
    main()
