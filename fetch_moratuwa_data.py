"""
Fetch real hourly weather & air-quality data for University of Moratuwa
from Open-Meteo (free, no API key), save CSV for training, and upload to Firebase.

Source: https://open-meteo.com/
  - Historical weather: ERA5 reanalysis (archive API)
  - Air quality: CAMS global atmospheric composition forecasts / reanalysis
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import config
import moratuwa_config as mc

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'


def _get(url, params):
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def fetch_weather_archive(start_date: str, end_date: str) -> pd.DataFrame:
    print(f'Fetching weather archive {start_date} → {end_date} …')
    payload = _get(mc.OPEN_METEO_ARCHIVE_URL, {
        'latitude': mc.MORATUWA_LAT,
        'longitude': mc.MORATUWA_LON,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(mc.WEATHER_HOURLY_VARS),
        'timezone': mc.MORATUWA_TIMEZONE,
    })
    hourly = payload['hourly']
    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly['time']),
        'temperature': hourly['temperature_2m'],
        'humidity': hourly['relative_humidity_2m'],
        'pressure': hourly['surface_pressure'],
        'rainfall': hourly['rain'],
        'wind_speed': hourly['wind_speed_10m'],
        'wind_direction': hourly['wind_direction_10m'],
        'cloud_cover': hourly['cloud_cover'],
        'light_intensity': hourly['shortwave_radiation'],
    })
    print(f'  ✓ {len(df)} weather hourly records')
    return df


def fetch_air_quality_archive(start_date: str, end_date: str) -> pd.DataFrame:
    print(f'Fetching air quality archive {start_date} → {end_date} …')
    payload = _get(mc.OPEN_METEO_AIR_QUALITY_URL, {
        'latitude': mc.MORATUWA_LAT,
        'longitude': mc.MORATUWA_LON,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(mc.AIR_QUALITY_HOURLY_VARS),
        'timezone': mc.MORATUWA_TIMEZONE,
    })
    hourly = payload['hourly']
    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly['time']),
        'pm10': hourly['pm10'],
        'pm2_5': hourly['pm2_5'],
        'aqi': hourly['us_aqi'],
        'european_aqi': hourly['european_aqi'],
    })
    print(f'  ✓ {len(df)} air quality hourly records')
    return df


def merge_datasets(weather_df: pd.DataFrame, aq_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(weather_df, aq_df, on='datetime', how='outer')
    merged = merged.sort_values('datetime').reset_index(drop=True)
    merged['voc'] = merged['pm2_5'] * 2.5  # proxy — not measured locally
    merged['co2'] = 410.0  # background atmospheric default
    merged['source'] = 'open-meteo'
    merged['location'] = mc.MORATUWA_LABEL
    return merged


def save_training_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.insert(0, 'Timestamp', out['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'))
    out = out.drop(columns=['datetime'])
    out.to_csv(path, index=False)
    print(f'✓ Saved training CSV: {path} ({len(out)} rows)')
    return path


def upload_to_firebase(df: pd.DataFrame, batch_size: int = 400) -> int:
    import firebase_admin
    from firebase_admin import credentials, db

    database_url, credentials_path = config.require_firebase_configuration()

    if not firebase_admin._apps:
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred, {'databaseURL': database_url})

    weather_updates = {}
    pollution_updates = {}
    uploaded = 0

    for _, row in df.iterrows():
        if pd.isna(row.get('temperature')):
            continue
        ts = int(row['datetime'].timestamp() * 1000)
        iso = row['datetime'].isoformat()

        weather_updates[f'weather_data/{ts}'] = {
            'id': ts,
            'timestamp': ts,
            'createdAt': iso,
            'temperature': _f(row['temperature']),
            'humidity': _f(row['humidity']),
            'pressure': _f(row['pressure']),
            'rainfall': _f(row.get('rainfall'), 0),
            'light': _f(row.get('light_intensity'), 0),
            'wind_speed': _f(row.get('wind_speed')),
            'source': 'open-meteo',
            'location': mc.MORATUWA_LABEL,
        }

        if not pd.isna(row.get('aqi')):
            pollution_updates[f'pollution_data/{ts}'] = {
                'id': ts,
                'timestamp': ts,
                'createdAt': iso,
                'aqi': _f(row['aqi']),
                'pm25': _f(row.get('pm2_5')),
                'pm10': _f(row.get('pm10')),
                'source': 'open-meteo',
                'location': mc.MORATUWA_LABEL,
            }
        uploaded += 1

    all_updates = {**weather_updates, **pollution_updates}
    keys = list(all_updates.keys())
    for i in range(0, len(keys), batch_size):
        chunk = {k: all_updates[k] for k in keys[i:i + batch_size]}
        db.reference('/').update(chunk)
        print(f'  Uploaded batch {i // batch_size + 1} ({len(chunk)} nodes)')

    meta = {
        'last_sync': datetime.utcnow().isoformat() + 'Z',
        'source': 'open-meteo',
        'location': mc.MORATUWA_LABEL,
        'latitude': mc.MORATUWA_LAT,
        'longitude': mc.MORATUWA_LON,
        'records': uploaded,
        'start': df['datetime'].min().isoformat(),
        'end': df['datetime'].max().isoformat(),
    }
    db.reference('/metadata/open_meteo_sync').set(meta)
    print(f'✓ Firebase upload complete ({uploaded} hourly records)')
    return uploaded


def _f(value, default=None):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return round(float(value), 2)


def resolve_date_range(start: str, end: str | None):
    start_date = start
    if end:
        end_date = end
    else:
        # Archive API is typically available through yesterday
        end_date = (datetime.now().date() - timedelta(days=1)).isoformat()
    return start_date, end_date


def main():
    parser = argparse.ArgumentParser(description='Fetch real Moratuwa weather/AQ data from Open-Meteo')
    parser.add_argument('--start', default=os.getenv('DATA_START', mc.DEFAULT_HISTORY_START))
    parser.add_argument('--end', default=os.getenv('DATA_END'))
    parser.add_argument('--csv', default=str(DATA_DIR / 'moratuwa_open_meteo.csv'))
    parser.add_argument('--firebase', action='store_true', help='Upload to Firebase Realtime Database')
    parser.add_argument('--skip-csv', action='store_true')
    args = parser.parse_args()

    start_date, end_date = resolve_date_range(args.start, args.end)
    print(f'\n=== Open-Meteo data pull: {mc.MORATUWA_LABEL} ===')
    print(f'Coordinates: {mc.MORATUWA_LAT}, {mc.MORATUWA_LON}')
    print(f'Period: {start_date} → {end_date}\n')

    weather_df = fetch_weather_archive(start_date, end_date)
    aq_df = fetch_air_quality_archive(start_date, end_date)
    merged = merge_datasets(weather_df, aq_df)

    if not args.skip_csv:
        save_training_csv(merged, Path(args.csv))

    if args.firebase:
        upload_to_firebase(merged)

    print('\nDone. Train with:')
    print(f'  TRAINING_CSV={args.csv} python train_lstm_model.py')


if __name__ == '__main__':
    main()
