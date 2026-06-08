from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

from train_aq_xgboost_model import AirQualityXGBoostPredictor

router = APIRouter(prefix="/predict", tags=["predictions"])
app = FastAPI(title="Weather & Air Quality Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
BEST_LSTM_MODEL = BASE_DIR / 'best_lstm_model.pth'


class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]


class WeatherPredictionPayload(BaseModel):
    temperature: float = Field(..., validation_alias='temp')
    humidity: float
    pressure: float
    aqi: float
    pm2_5: float = Field(..., validation_alias='pm25')
    light: float
    datetime: Optional[datetime] = None
    timestamp: Optional[float] = None

    model_config = {
        'populate_by_name': True,
        'extra': 'ignore',
    }


class WeatherPredictionService:
    FEATURE_ORDER = ['temperature', 'humidity', 'pressure', 'aqi', 'pm2_5', 'light']
    LOOKBACK_STEPS = 24

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler_X_mean = None
        self.scaler_X_scale = None
        self.scaler_y_mean = None
        self.scaler_y_scale = None
        self.load_model()

    def load_model(self) -> None:
        model_path = self.model_dir / 'lstm_weather_model.keras'
        scaler_path = self.model_dir / 'lstm_scalers.npz'

        if not self.model_dir.exists() or not self.model_dir.is_dir():
            raise FileNotFoundError(f'Model directory not found: {self.model_dir}')
        if not model_path.exists():
            raise FileNotFoundError(f'Weather model file not found: {model_path}')
        if not scaler_path.exists():
            raise FileNotFoundError(f'Scaler file not found: {scaler_path}')

        try:
            self.model = load_model(str(model_path))
        except Exception as exc:
            raise RuntimeError(f'Failed to load Keras weather LSTM model: {exc}') from exc

        try:
            scaler_data = np.load(str(scaler_path))
            self.scaler_X_mean = scaler_data['scaler_X_mean']
            self.scaler_X_scale = scaler_data['scaler_X_scale']
            self.scaler_y_mean = scaler_data['scaler_y_mean']
            self.scaler_y_scale = scaler_data['scaler_y_scale']
        except Exception as exc:
            raise RuntimeError(f'Failed to load weather scaler data: {exc}') from exc

    def _prepare_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not isinstance(records, list) or len(records) == 0:
            raise HTTPException(status_code=400, detail='records must be a non-empty list')

        df = pd.DataFrame(records)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        else:
            df['datetime'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='H')

        if df['datetime'].isna().any():
            raise HTTPException(status_code=400, detail='datetime values must be parseable')

        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    def predict(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError('Weather model is not loaded')

        df = self._prepare_dataframe(records)

        for feature in self.FEATURE_ORDER:
            if feature not in df.columns:
                raise HTTPException(status_code=400, detail=f'Missing required feature: {feature}')

        X_seq = df[self.FEATURE_ORDER].values
        if len(X_seq) < self.LOOKBACK_STEPS:
            pad_count = self.LOOKBACK_STEPS - len(X_seq)
            pad_rows = np.repeat(X_seq[[0]], pad_count, axis=0)
            X_seq = np.vstack([pad_rows, X_seq])

        X_seq = X_seq[-self.LOOKBACK_STEPS:]
        predictions = self._predict_sequence(X_seq)

        return {
            'model_type': 'weather',
            'predictions': predictions
        }

    def _predict_sequence(self, X_seq: np.ndarray) -> Dict[str, float]:
        X_scaled = (X_seq - self.scaler_X_mean) / self.scaler_X_scale
        X_scaled = X_scaled.reshape(1, self.LOOKBACK_STEPS, len(self.FEATURE_ORDER))

        predictions_scaled = self.model.predict(X_scaled, verbose=0)
        predictions = predictions_scaled[0] * self.scaler_y_scale + self.scaler_y_mean

        return {
            feature: float(predictions[idx])
            for idx, feature in enumerate(self.FEATURE_ORDER)
        }

    def forecast(self, records: List[Dict[str, Any]], hours: int = 24) -> List[Dict[str, Any]]:
        df = self._prepare_dataframe(records)

        for feature in self.FEATURE_ORDER:
            if feature not in df.columns:
                raise HTTPException(status_code=400, detail=f'Missing required feature: {feature}')

        X_seq = df[self.FEATURE_ORDER].values
        if len(X_seq) < self.LOOKBACK_STEPS:
            pad_count = self.LOOKBACK_STEPS - len(X_seq)
            pad_rows = np.repeat(X_seq[[0]], pad_count, axis=0)
            X_seq = np.vstack([pad_rows, X_seq])

        X_seq = X_seq[-self.LOOKBACK_STEPS:].astype(float)
        last_datetime = df['datetime'].iloc[-1]

        forecast_points: List[Dict[str, Any]] = []
        for step in range(1, hours + 1):
            predicted_values = self._predict_sequence(X_seq)
            forecast_point = {
                'datetime': (last_datetime + pd.Timedelta(hours=step)).isoformat(),
                **predicted_values
            }
            forecast_points.append(forecast_point)

            X_seq = np.vstack([
                X_seq[1:],
                np.array([predicted_values[feature] for feature in self.FEATURE_ORDER], dtype=float)
            ])

        return forecast_points


def aqi_to_label(aqi: float) -> str:
    if aqi <= 50:
        return 'Good'
    if aqi <= 100:
        return 'Moderate'
    if aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    if aqi <= 200:
        return 'Unhealthy'
    if aqi <= 300:
        return 'Very Unhealthy'
    return 'Hazardous'


class AirQualityPredictionService:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.predictor = AirQualityXGBoostPredictor()
        self.predictor.load_latest_models(output_dir=str(self.model_dir))

    def _prepare_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not isinstance(records, list) or len(records) == 0:
            raise HTTPException(status_code=400, detail='records must be a non-empty list')

        df = pd.DataFrame(records)
        if 'datetime' not in df.columns:
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            else:
                raise HTTPException(status_code=400, detail='Each record must include datetime or timestamp')
        else:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        if df['datetime'].isna().any():
            raise HTTPException(status_code=400, detail='datetime values must be parseable')

        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df

    def _build_feature_vector(self, df: pd.DataFrame) -> np.ndarray:
        feature_columns = self.predictor.feature_columns
        lookback_length = self.predictor.lookback_hours * 60
        for feature in feature_columns:
            if feature not in df.columns:
                df[feature] = 0.0

        X_seq = df[feature_columns].values
        if len(X_seq) < lookback_length:
            raise HTTPException(
                status_code=400,
                detail=(
                    f'Air quality prediction requires at least {lookback_length} rows, '
                    f'but received {len(X_seq)}'
                )
            )

        X_seq = X_seq[-lookback_length:]
        features = []
        for col_idx, col in enumerate(feature_columns):
            col_data = X_seq[:, col_idx]
            features.extend([
                float(col_data[-1]),
                float(np.mean(col_data)),
                float(np.std(col_data)),
                float(col_data[-1] - col_data[0]),
                float(np.max(col_data)),
                float(np.min(col_data))
            ])

        return np.array(features, dtype=float).reshape(1, -1)

    def predict(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = self._prepare_dataframe(records)
        df = self._add_time_features(df)
        X_input = self._build_feature_vector(df)

        predictions: Dict[str, float] = {}
        for param, model in self.predictor.models.items():
            scaler = self.predictor.scalers.get(param)
            if scaler is None:
                continue
            X_scaled = scaler.transform(X_input)
            pred = model.predict(X_scaled)
            predictions[param] = float(pred[0])

        return {
            'model_type': 'air_quality',
            'forecast_horizon_hours': int(self.predictor.forecast_horizon),
            'predictions': predictions
        }


try:
    weather_service = WeatherPredictionService()
except Exception as exc:
    weather_service = None
    load_weather_error = str(exc)

try:
    air_quality_service = AirQualityPredictionService()
except Exception as exc:
    air_quality_service = None
    load_aq_error = str(exc)


@router.post('/weather')
async def predict_weather(payload: WeatherPredictionPayload) -> Dict[str, Any]:
    print('--- RECEIVED PAYLOAD FROM FRONTEND ---', payload)
    if weather_service is None:
        raise HTTPException(status_code=503, detail=f'Weather model unavailable: {load_weather_error}')

    record = payload.model_dump(by_alias=False, exclude_none=True)
    try:
        predictions = weather_service.predict([record])
    except Exception as e:
        print('LOGGED ERROR:', str(e))
        raise HTTPException(status_code=400, detail=str(e))

    forecast = weather_service.forecast([record], hours=24)
    return {
        'model_type': predictions.get('model_type', 'weather'),
        'temperature': predictions['predictions']['temperature'],
        'humidity': predictions['predictions']['humidity'],
        'pressure': predictions['predictions']['pressure'],
        'aqi': predictions['predictions']['aqi'],
        'pm25': predictions['predictions']['pm2_5'],
        'light': predictions['predictions']['light'],
        'predicted_temperature': predictions['predictions']['temperature'],
        'predicted_aqi': predictions['predictions']['aqi'],
        'predicted_pm25': predictions['predictions']['pm2_5'],
        'predicted_air_quality': aqi_to_label(predictions['predictions']['aqi']),
        'forecast_horizon_hours': 24,
        'hourly_forecast': forecast,
        'forecast_series': {
            feature: [point[feature] for point in forecast]
            for feature in weather_service.FEATURE_ORDER
        },
    }


@router.post('/air-quality')
async def predict_air_quality(request: PredictionRequest) -> Dict[str, Any]:
    if air_quality_service is None:
        raise HTTPException(status_code=503, detail=f'Air quality model unavailable: {load_aq_error}')

    response = air_quality_service.predict(request.records)
    return response


@router.get('/health')
async def health_check() -> Dict[str, Any]:
    return {
        'status': 'healthy',
        'weather_model_loaded': weather_service is not None,
        'air_quality_model_loaded': air_quality_service is not None,
        'errors': {
            'weather': None if weather_service is not None else load_weather_error,
            'air_quality': None if air_quality_service is not None else load_aq_error
        }
    }

app.include_router(router)
