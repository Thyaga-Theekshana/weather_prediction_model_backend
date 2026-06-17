"""Open-Meteo API client for live 7-day forecasts near University of Moratuwa."""

from datetime import datetime

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
