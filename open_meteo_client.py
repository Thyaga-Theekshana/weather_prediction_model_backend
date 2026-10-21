"""Open-Meteo API client for live 7-day forecasts near University of Moratuwa."""

from datetime import datetime, timedelta

import numpy as np
import requests

import moratuwa_config as mc

# WMO weather interpretation codes → frontend icon keys
WMO_TO_FORECAST = {
    0: {'icon': 'fa-sun', 'type': 'sun', 'desc': 'Clear sky'},
    1: {'icon': 'fa-sun', 'type': 'sun', 'desc': 'Mainly clear'},
    2: {'icon': 'fa-cloud-sun', 'type': 'cloud-sun', 'desc': 'Partly cloudy'},
    3: {'icon': 'fa-cloud', 'type': 'cloud', 'desc': 'Overcast'},
    45: {'icon': 'fa-smog', 'type': 'cloud', 'desc': 'Fog'},
    48: {'icon': 'fa-smog', 'type': 'cloud', 'desc': 'Depositing rime fog'},
    51: {'icon': 'fa-cloud-rain', 'type': 'rain', 'desc': 'Light drizzle'},
    53: {'icon': 'fa-cloud-rain', 'type': 'rain', 'desc': 'Drizzle'},
    55: {'icon': 'fa-cloud-showers-heavy', 'type': 'rain', 'desc': 'Heavy drizzle'},
    61: {'icon': 'fa-cloud-rain', 'type': 'rain', 'desc': 'Light rain'},
    63: {'icon': 'fa-cloud-showers-heavy', 'type': 'rain', 'desc': 'Moderate rain'},
    65: {'icon': 'fa-cloud-showers-heavy', 'type': 'rain', 'desc': 'Heavy rain'},
    80: {'icon': 'fa-cloud-showers-heavy', 'type': 'rain', 'desc': 'Rain showers'},
    81: {'icon': 'fa-cloud-showers-heavy', 'type': 'rain', 'desc': 'Heavy showers'},
    82: {'icon': 'fa-cloud-showers-heavy', 'type': 'rain', 'desc': 'Violent showers'},
    95: {'icon': 'fa-bolt', 'type': 'storm', 'desc': 'Thunderstorm'},
    96: {'icon': 'fa-bolt', 'type': 'storm', 'desc': 'Thunderstorm with hail'},
    99: {'icon': 'fa-bolt', 'type': 'storm', 'desc': 'Heavy thunderstorm'},
}


def _wmo_info(code: int) -> dict:
    if code in WMO_TO_FORECAST:
        return WMO_TO_FORECAST[code]
    if code in range(56, 68):
        return WMO_TO_FORECAST.get(63, {'icon': 'fa-cloud-rain', 'type': 'rain', 'desc': 'Rain'})
    if code in range(71, 78):
        return {'icon': 'fa-snowflake', 'type': 'cloud', 'desc': 'Snow'}
    return {'icon': 'fa-cloud-sun', 'type': 'cloud-sun', 'desc': 'Mixed conditions'}


def _day_name(date_str: str, index: int) -> str:
    if index == 0:
        return 'Today'
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.strftime('%A')


def fetch_7day_forecast() -> dict:
    """Return 7-day daily forecast from Open-Meteo (no API key required)."""
    weather = requests.get(mc.OPEN_METEO_FORECAST_URL, params={
        'latitude': mc.MORATUWA_LAT,
        'longitude': mc.MORATUWA_LON,
        'daily': ','.join(mc.DAILY_FORECAST_VARS),
        'forecast_days': 7,
        'timezone': mc.MORATUWA_TIMEZONE,
    }, timeout=30)
    weather.raise_for_status()
    weather_json = weather.json()

    aq_daily_aqi = [None] * 7
    try:
        aq = requests.get(mc.OPEN_METEO_AIR_QUALITY_URL, params={
            'latitude': mc.MORATUWA_LAT,
            'longitude': mc.MORATUWA_LON,
            'hourly': 'us_aqi,pm2_5',
            'forecast_days': 7,
            'timezone': mc.MORATUWA_TIMEZONE,
        }, timeout=30)
        aq.raise_for_status()
        aq_json = aq.json()
        aq_times = aq_json['hourly']['time']
        aq_values = aq_json['hourly']['us_aqi']
        daily = {}
        for t, aqi in zip(aq_times, aq_values):
            day = t[:10]
            if aqi is not None:
                daily.setdefault(day, []).append(aqi)
        for i, day in enumerate(weather_json['daily']['time']):
            vals = daily.get(day, [])
            aq_daily_aqi[i] = round(sum(vals) / len(vals)) if vals else None
    except Exception:
        pass

    daily = weather_json['daily']
    days = []
    for i, date_str in enumerate(daily['time']):
        code = daily['weather_code'][i]
        info = _wmo_info(code)
        days.append({
            'day': _day_name(date_str, i),
            'date': date_str,
            'dateLabel': datetime.strptime(date_str, '%Y-%m-%d').strftime('%b %d'),
            'icon': info['icon'],
            'type': info['type'],
            'desc': info['desc'],
            'weather_code': code,
            'max': round(daily['temperature_2m_max'][i]),
            'min': round(daily['temperature_2m_min'][i]),
            'precipitation_mm': daily['precipitation_sum'][i],
            'wind_max_kmh': daily['wind_speed_10m_max'][i],
            'uv_max': daily['uv_index_max'][i],
            'aqi': aq_daily_aqi[i],
        })

    return {
        'source': 'open-meteo',
        'provider': 'https://open-meteo.com/',
        'location': mc.MORATUWA_LABEL,
        'latitude': mc.MORATUWA_LAT,
        'longitude': mc.MORATUWA_LON,
        'timezone': mc.MORATUWA_TIMEZONE,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'days': days,
    }


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
