"""Open-Meteo API client for hourly forecasts near University of Moratuwa."""

from datetime import datetime

import numpy as np
import requests

import moratuwa_config as mc

# ─── 24-hour helpers ───────────────────────────────────────────────────────────

def fetch_24h_seed_data() -> list:
    """Return the last 24 hours of real measured hourly data for LSTM seeding.

    The list has up to 24 dicts, each matching the LSTM feature column names:
      temperature, humidity, pressure, rainfall, light_intensity,
      wind_speed, wind_direction, voc, pm2_5, co2,
      hour_sin, hour_cos, month_sin, month_cos
    Sorted oldest-first so the model can use the whole list as the lookback window.
    """
    hourly_vars = [
        'temperature_2m', 'relative_humidity_2m', 'surface_pressure',
        'rain', 'wind_speed_10m', 'wind_direction_10m', 'shortwave_radiation',
    ]
    weather_resp = requests.get(mc.OPEN_METEO_FORECAST_URL, params={
        'latitude': mc.MORATUWA_LAT,
        'longitude': mc.MORATUWA_LON,
        'hourly': ','.join(hourly_vars),
        'past_hours': 24,
        'forecast_hours': 1,
        'wind_speed_unit': 'ms',
        'timezone': mc.MORATUWA_TIMEZONE,
    }, timeout=30)
    weather_resp.raise_for_status()
    wj = weather_resp.json()

    # Fetch AQ data (pm2_5)
    total_len = len(wj['hourly']['time'])
    pm25_vals = [15.0] * total_len
    try:
        aq_resp = requests.get(mc.OPEN_METEO_AIR_QUALITY_URL, params={
            'latitude': mc.MORATUWA_LAT,
            'longitude': mc.MORATUWA_LON,
            'hourly': 'pm2_5',
            'past_hours': 24,
            'forecast_hours': 1,
            'timezone': mc.MORATUWA_TIMEZONE,
        }, timeout=30)
        aq_resp.raise_for_status()
        aq_vals = aq_resp.json()['hourly'].get('pm2_5', [])
        pm25_vals = [v if v is not None else 15.0 for v in aq_vals[:total_len]]
    except Exception:
        pass

    times     = wj['hourly']['time']
    temps     = wj['hourly']['temperature_2m']
    hums      = wj['hourly']['relative_humidity_2m']
    press     = wj['hourly']['surface_pressure']
    rains     = wj['hourly']['rain']
    winds     = wj['hourly']['wind_speed_10m']
    wind_dirs = wj['hourly']['wind_direction_10m']
    radiation = wj['hourly']['shortwave_radiation']

    # Keep only the last 24 entries (drop the +1 forecast_hours tail)
    seed_times = times[-24:]
    records = []
    for i, t_str in enumerate(seed_times):
        raw_i = total_len - 24 + i  # absolute index into the API arrays
        dt = datetime.fromisoformat(t_str)
        hour = dt.hour
        month = dt.month

        def _safe(arr, idx, default):
            try:
                v = arr[idx]
                return float(v) if v is not None else default
            except (IndexError, TypeError):
                return default

        records.append({
            'datetime':       dt.isoformat(),
            'temperature':    _safe(temps,     raw_i, 28.0),
            'humidity':       _safe(hums,      raw_i, 70.0),
            'pressure':       _safe(press,     raw_i, 1010.0),
            'rainfall':       _safe(rains,     raw_i, 0.0),
            'light_intensity':_safe(radiation, raw_i, 200.0),
            'wind_speed':     _safe(winds,     raw_i, 5.0),
            'wind_direction': _safe(wind_dirs, raw_i, 180.0),
            'voc':   100.0,   # no VOC from Open-Meteo; use neutral placeholder
            'pm2_5': _safe(pm25_vals, raw_i, 15.0),
            'co2':   400.0,   # atmospheric baseline
            'hour_sin':  float(np.sin(2 * np.pi * hour / 24)),
            'hour_cos':  float(np.cos(2 * np.pi * hour / 24)),
            'month_sin': float(np.sin(2 * np.pi * month / 12)),
            'month_cos': float(np.cos(2 * np.pi * month / 12)),
        })

    return records


def fetch_24h_hourly_forecast() -> list:
    """Return the next 24 hours of Open-Meteo hourly forecast as a list of dicts.

    Used as a fallback when the ML model is unavailable.
    Each dict has keys: timeLabel, datetime, temperature, humidity,
                        pressure, wind_speed, rainfall, aqi.
    """
    hourly_vars = [
        'temperature_2m', 'relative_humidity_2m', 'surface_pressure',
        'rain', 'wind_speed_10m',
    ]
    resp = requests.get(mc.OPEN_METEO_FORECAST_URL, params={
        'latitude': mc.MORATUWA_LAT,
        'longitude': mc.MORATUWA_LON,
        'hourly': ','.join(hourly_vars),
        'forecast_hours': 24,
        'wind_speed_unit': 'ms',
        'timezone': mc.MORATUWA_TIMEZONE,
    }, timeout=30)
    resp.raise_for_status()
    j = resp.json()

    # Try to get AQI
    aqi_vals = [None] * 24
    try:
        aq = requests.get(mc.OPEN_METEO_AIR_QUALITY_URL, params={
            'latitude': mc.MORATUWA_LAT,
            'longitude': mc.MORATUWA_LON,
            'hourly': 'us_aqi',
            'forecast_hours': 24,
            'timezone': mc.MORATUWA_TIMEZONE,
        }, timeout=30)
        aq.raise_for_status()
        raw = aq.json()['hourly'].get('us_aqi', [])
        aqi_vals = [v for v in raw[:24]]
    except Exception:
        pass

    hours = j['hourly']
    results = []
    for i, t_str in enumerate(hours['time'][:24]):
        dt = datetime.fromisoformat(t_str)
        aqi = aqi_vals[i] if i < len(aqi_vals) and aqi_vals[i] is not None else None
        results.append({
            'timeLabel': dt.strftime('%H:%M'),
            'datetime': dt.isoformat(),
            'temperature': round(float(hours['temperature_2m'][i] or 28), 1),
            'humidity': round(float(hours['relative_humidity_2m'][i] or 70), 1),
            'pressure': round(float(hours['surface_pressure'][i] or 1010), 1),
            'wind_speed': round(float(hours['wind_speed_10m'][i] or 5), 1),
            'rainfall': round(float(hours['rain'][i] or 0), 2),
            'aqi': round(float(aqi)) if aqi is not None else None,
            'source': 'open-meteo',
        })

    return results
