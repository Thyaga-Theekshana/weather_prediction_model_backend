"""University of Moratuwa, Sri Lanka — Open-Meteo location constants."""

MORATUWA_LAT = 6.7956
MORATUWA_LON = 79.9009
MORATUWA_TIMEZONE = 'Asia/Colombo'
MORATUWA_LABEL = 'University of Moratuwa'

# Historical pull window (override via env in fetch script)
DEFAULT_HISTORY_START = '2026-04-01'

OPEN_METEO_ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'
OPEN_METEO_FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
OPEN_METEO_AIR_QUALITY_URL = 'https://air-quality-api.open-meteo.com/v1/air-quality'

WEATHER_HOURLY_VARS = [
    'temperature_2m',
    'relative_humidity_2m',
    'surface_pressure',
    'rain',
    'wind_speed_10m',
    'wind_direction_10m',
    'cloud_cover',
    'shortwave_radiation',
]

AIR_QUALITY_HOURLY_VARS = [
    'pm10',
    'pm2_5',
    'us_aqi',
    'european_aqi',
]

DAILY_FORECAST_VARS = [
    'weather_code',
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_sum',
    'wind_speed_10m_max',
    'uv_index_max',
]
