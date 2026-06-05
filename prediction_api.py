from typing import Any, Dict, List
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from train_aq_xgboost_model import AirQualityXGBoostPredictor
from train_lstm_model import WeatherLSTM

router = APIRouter(prefix="/predict", tags=["predictions"])
app = FastAPI(title="Weather & Air Quality Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
BEST_LSTM_MODEL = BASE_DIR / 'best_lstm_model.pth'


class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]


class WeatherPredictionService:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metadata = None
        self.load_model()

    def _find_latest_file(self, prefix: str, suffix: str) -> Path:
        candidates = [p for p in self.model_dir.glob(f'{prefix}*{suffix}') if p.is_file()]
        if not candidates:
            raise FileNotFoundError(f'No files found matching {prefix}*{suffix} in {self.model_dir}')
        return sorted(candidates)[-1]

    def load_model(self) -> None:
        if self.model_dir.exists() and self.model_dir.is_dir():
            try:
                if BEST_LSTM_MODEL.exists():
                    latest_metadata = self._find_latest_file('lstm_metadata_', '.json')
                    model_path = BEST_LSTM_MODEL
                else:
                    latest_model = self._find_latest_file('lstm_weather_', '.pth')
                    timestamp = latest_model.stem.replace('lstm_weather_', '')
                    latest_metadata = self.model_dir / f'lstm_metadata_{timestamp}.json'
                    model_path = latest_model

                with latest_metadata.open('r', encoding='utf-8') as f:
                    self.metadata = json.load(f)

                timestamp = self.metadata.get('timestamp')
                if not timestamp:
                    raise ValueError('LSTM metadata is missing timestamp')

                self.scaler_X = joblib.load(self.model_dir / f'scaler_X_{timestamp}.pkl')
                self.scaler_y = joblib.load(self.model_dir / f'scaler_y_{timestamp}.pkl')

                self.model = WeatherLSTM(
                    input_size=self.metadata['input_size'],
                    hidden_size=128,
                    num_layers=2,
                    output_size=self.metadata['output_size']
                )
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
            except Exception as exc:
                raise RuntimeError(f'Failed to load weather LSTM model: {exc}')
        else:
            raise FileNotFoundError(f'Model directory not found: {self.model_dir}')

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

    def predict(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError('Weather model is not loaded')

        df = self._prepare_dataframe(records)
        df = self._add_time_features(df)

        feature_columns = self.metadata['feature_columns']
        expected_length = self.metadata['lookback_hours'] * 60

        for feature in feature_columns:
            if feature not in df.columns:
                df[feature] = 0.0

        X_seq = df[feature_columns].values
        if len(X_seq) < expected_length:
            raise HTTPException(
                status_code=400,
                detail=(
                    f'Weather prediction requires at least {expected_length} rows, '
                    f'but received {len(X_seq)}'
                )
            )

        X_seq = X_seq[-expected_length:]
        X_scaled = self.scaler_X.transform(X_seq)
        X_scaled = X_scaled.reshape(1, expected_length, len(feature_columns))

        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(X_scaled)).numpy()

        predictions = self.scaler_y.inverse_transform(predictions)
        return {
            'model_type': 'weather',
            'forecast_horizon_hours': int(self.metadata['forecast_horizon']),
            'predictions': {
                param: float(predictions[0, idx])
                for idx, param in enumerate(self.metadata['target_params'])
            }
        }


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
async def predict_weather(request: PredictionRequest) -> Dict[str, Any]:
    if weather_service is None:
        raise HTTPException(status_code=503, detail=f'Weather model unavailable: {load_weather_error}')

    response = weather_service.predict(request.records)
    return response


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
