"""
Weather Prediction API
Flask server to serve weather predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
import os

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for ML path
    torch = None

import config

import open_meteo_client

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ["http://127.0.0.1:8080", "http://localhost:8080", "http://127.0.0.1:8000", "http://localhost:8000"]}},
    supports_credentials=True,
)

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'moratuwa_sensors_20k_dataset.csv')


def load_csv_sensor_fallback(limit=48):
    """Load recent rows from the local Moratuwa CSV dataset for offline fallback."""
    if not os.path.exists(DATASET_PATH):
        return []

    try:
        df = pd.read_csv(DATASET_PATH)
        if df.empty:
            return []

        timestamp_col = 'Timestamp' if 'Timestamp' in df.columns else None
        if timestamp_col is None:
            return []

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
        if df.empty:
            return []

        subset = df.tail(min(limit, len(df))).copy()
        records = []
        for _, row in subset.iterrows():
            try:
                pm25 = float(row.get('PM2_5', np.nan)) if 'PM2_5' in row.index else np.nan
                pm10 = float(row.get('PM10', np.nan)) if 'PM10' in row.index else np.nan
                aqi = int(round(max(5, min(300, (pm25 * 3) + 10)))) if not np.isnan(pm25) else 42

                record = {
                    'timestamp': int(pd.Timestamp(row[timestamp_col]).timestamp() * 1000),
                    'datetime': pd.Timestamp(row[timestamp_col]).isoformat(),
                    'temperature': round(float(row.get('Temperature', np.nan)), 2) if 'Temperature' in row.index else None,
                    'humidity': round(float(row.get('Humidity', np.nan)), 2) if 'Humidity' in row.index else None,
                    'pressure': round(float(row.get('Pressure', np.nan)), 2) if 'Pressure' in row.index else None,
                    'rainfall': round(float(row.get('Rainfall', 0.0)), 2) if 'Rainfall' in row.index else 0.0,
                    'light': round(float(row.get('Light_Intensity', np.nan)), 2) if 'Light_Intensity' in row.index else None,
                    'light_intensity': round(float(row.get('Light_Intensity', np.nan)), 2) if 'Light_Intensity' in row.index else None,
                    'wind_speed': round(float(row.get('Wind_Speed', np.nan)), 2) if 'Wind_Speed' in row.index else None,
                    'wind_direction': row.get('Wind_Direction') if 'Wind_Direction' in row.index else None,
                    'aqi': aqi,
                    'pm25': round(float(pm25), 2) if not np.isnan(pm25) else None,
                    'pm10': round(float(pm10), 2) if not np.isnan(pm10) else None,
                    'pm2_5': round(float(pm25), 2) if not np.isnan(pm25) else None,
                }
                records.append(record)
            except Exception:
                continue

        return records
    except Exception as e:
        print(f'CSV fallback load error: {e}')
        return []


class PredictionService:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metadata = None
        self.model_type = 'lstm'  # or 'xgboost'
        
        self.load_latest_model()
    
    def load_latest_model(self):
        """Load the most recent trained model"""
        try:
            if not os.path.isdir(self.model_dir):
                print(f"Model directory not found: {self.model_dir}")
                return False

            # Find latest LSTM checkpoint (.pth)
            lstm_files = [f for f in os.listdir(self.model_dir) if f.startswith('lstm_weather_') and f.endswith('.pth')]
            if not lstm_files:
                print(f"No trained .pth models in {self.model_dir}. Run: python train_lstm_model.py")
                return False
            latest_lstm = sorted(lstm_files)[-1]
            timestamp = latest_lstm[len('lstm_weather_'):].replace('.pth', '')

            model_path = f'{self.model_dir}/{latest_lstm}'
            scaler_X_path = f'{self.model_dir}/scaler_X_{timestamp}.pkl'
            scaler_y_path = f'{self.model_dir}/scaler_y_{timestamp}.pkl'
            metadata_path = f'{self.model_dir}/lstm_metadata_{timestamp}.json'

            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            self.scaler_X = joblib.load(scaler_X_path)
            self.scaler_y = joblib.load(scaler_y_path)

            if torch is None:
                print("PyTorch is not available; falling back to Open-Meteo forecasts")
                return False

            from train_lstm_model import WeatherLSTM
            device = torch.device('cpu')
            self.model = WeatherLSTM(
                input_size=self.metadata['input_size'],
                hidden_size=128,
                num_layers=2,
                output_size=self.metadata['output_size']
            ).to(device)

            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()

            self.model_type = 'lstm'
            print(f"✓ Loaded LSTM model from {timestamp}")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _ensure_datetime(self, df):
        if 'datetime' not in df.columns:
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                raise ValueError('Missing datetime or timestamp column')
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def _add_time_features(self, df):
        df = self._ensure_datetime(df)
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df

    def _prepare_dataframe(self, data):
        if isinstance(data, dict):
            records = []
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                row = dict(value)
                if str(key).isdigit():
                    row['timestamp'] = int(key)
                row['datetime'] = row.get('datetime') or row.get('timestamp')
                records.append(row)
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame(data)

        df = self._ensure_datetime(df)
        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    def predict(self, recent_data):
        """
        Make prediction for next 48 hours
        
        Args:
            recent_data: DataFrame with last 24 hours of sensor readings
        
        Returns:
            Dictionary with predictions for each parameter
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        try:
            # Prepare features
            feature_cols = self.metadata['feature_columns']
            X_seq = recent_data[feature_cols].values
            
            # Ensure correct sequence length (hourly steps)
            expected_length = self.metadata['lookback_hours']
            if len(X_seq) < expected_length:
                # Pad with last value if needed
                padding = np.repeat(X_seq[-1:], expected_length - len(X_seq), axis=0)
                X_seq = np.vstack([padding, X_seq])
            elif len(X_seq) > expected_length:
                X_seq = X_seq[-expected_length:]
            
            # Normalize
            X_scaled = self.scaler_X.transform(X_seq)
            X_scaled = X_scaled.reshape(1, expected_length, len(feature_cols))
            
            # Predict
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                predictions = self.model(X_tensor).numpy()
            
            # Inverse transform
            predictions = self.scaler_y.inverse_transform(predictions)
            
            # Format results
            result = {}
            for i, param in enumerate(self.metadata['target_params']):
                result[param] = float(predictions[0, i])
            
            result['forecast_time'] = datetime.now().isoformat()
            result['forecast_horizon_hours'] = self.metadata['forecast_horizon']
            
            return result
        
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

    def predict_24h_hourly(self, seed_records):
        """
        Generate 24 individual hourly predictions using autoregressive LSTM inference.

        Args:
            seed_records: list of dicts (last 24h of hourly data), each with keys matching
                          the model's feature_columns (e.g. temperature, humidity, etc.)

        Returns:
            list of 24 dicts, each with timeLabel + predicted params + estimated aqi
        """
        if self.model is None:
            raise ValueError("No model loaded")

        if torch is None:
            raise ValueError("PyTorch is not available")

        import math
        feature_cols = self.metadata['feature_columns']
        target_params = self.metadata['target_params']
        lookback = self.metadata['lookback_hours']  # 24

        # Build seed DataFrame
        df = pd.DataFrame(seed_records)
        df = self._add_time_features(df) if 'hour_sin' not in df.columns else df

        # Fill missing feature columns with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_cols].fillna(0.0)

        # Pad or trim to exactly lookback length
        X = df.values
        if len(X) < lookback:
            padding = np.repeat(X[:1], lookback - len(X), axis=0)
            X = np.vstack([padding, X])
        elif len(X) > lookback:
            X = X[-lookback:]

        # We'll slide a rolling window through 24 autoregressive steps
        # Current window shape: (lookback, n_features)
        window = X.copy().astype(np.float32)

        hourly_results = []
        now = datetime.now()

        # Indices of target params in feature_cols (for rolling-window update)
        target_indices = {param: feature_cols.index(param)
                         for param in target_params if param in feature_cols}

        for step in range(24):
            forecast_dt = now + timedelta(hours=step + 1)
            hour = forecast_dt.hour
            month = forecast_dt.month

            # Scale current window
            window_scaled = self.scaler_X.transform(window)
            X_tensor = torch.FloatTensor(window_scaled.reshape(1, lookback, len(feature_cols)))

            with torch.no_grad():
                pred_scaled = self.model(X_tensor).numpy()  # (1, n_targets)

            pred = self.scaler_y.inverse_transform(pred_scaled)[0]  # (n_targets,)

            # Build result dict for this hour
            step_result = {
                'timeLabel': forecast_dt.strftime('%H:%M'),
                'datetime': forecast_dt.isoformat(),
                'source': 'lstm',
            }
            for i, param in enumerate(target_params):
                step_result[param] = round(float(pred[i]), 2)

            # Estimate AQI from predicted weather
            wind = step_result.get('wind_speed', 5.0)
            rain = step_result.get('rainfall', 0.0)
            traffic = 15 if (7 <= hour <= 9 or 17 <= hour <= 19) else (8 if hour >= 21 or hour <= 4 else -5)
            dispersion = math.exp(-0.03 * max(wind, 0.1))
            rain_wash = -12 if rain > 0.5 else 0
            aqi_est = max(5, round(45 * dispersion + traffic * dispersion + rain_wash +
                                   4 * math.sin(2 * math.pi * hour / 24)))
            step_result['aqi'] = aqi_est

            hourly_results.append(step_result)

            # Slide window: drop oldest row, append new row using predicted values
            new_row = window[-1].copy()
            for param, idx in target_indices.items():
                param_idx = target_params.index(param)
                new_row[idx] = float(pred[param_idx])
            # Update time cyclical features for the next step hour
            next_hour = (hour + 1) % 24
            next_month = month
            if 'hour_sin' in feature_cols:
                new_row[feature_cols.index('hour_sin')] = math.sin(2 * math.pi * next_hour / 24)
            if 'hour_cos' in feature_cols:
                new_row[feature_cols.index('hour_cos')] = math.cos(2 * math.pi * next_hour / 24)
            if 'month_sin' in feature_cols:
                new_row[feature_cols.index('month_sin')] = math.sin(2 * math.pi * next_month / 12)
            if 'month_cos' in feature_cols:
                new_row[feature_cols.index('month_cos')] = math.cos(2 * math.pi * next_month / 12)

            window = np.vstack([window[1:], new_row.reshape(1, -1)])

        return hourly_results


# Initialize service
prediction_service = PredictionService()

@app.route('/', methods=['GET'])
def home():
    """API home page"""
    return jsonify({
        'service': 'Weather Prediction API',
        'version': '2.0',
        'model_type': prediction_service.model_type,
        'endpoints': {
            '/predict': 'POST - Single-step weather prediction from supplied data',
            '/forecast/24h': 'GET - ML-powered 24-hour hourly forecast (LSTM + Open-Meteo seed)',
            '/forecast/7day': 'GET - Open-Meteo 7-day forecast (Moratuwa)',
            '/health': 'GET - Check API health',
            '/model-info': 'GET - Get model information'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': prediction_service.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if prediction_service.metadata is None:
        return jsonify({'error': 'No model loaded'}), 500
    
    return jsonify({
        'model_type': prediction_service.model_type,
        'metadata': prediction_service.metadata
    })

@app.route('/forecast/24h', methods=['GET'])
def forecast_24h():
    """ML-powered 24-hour hourly forecast.

    Strategy:
    1. Fetch last 24h of real weather data from Open-Meteo as LSTM seed.
    2. Run predict_24h_hourly() autoregressively (LSTM → 24 hourly steps).
    3. If the model is not loaded, fall back to direct Open-Meteo hourly forecast.
    """
    # Try ML path first
    if prediction_service.model is not None:
        try:
            seed = open_meteo_client.fetch_24h_seed_data()
            hourly = prediction_service.predict_24h_hourly(seed)
            return jsonify({
                'success': True,
                'source': 'lstm',
                'model_timestamp': prediction_service.metadata.get('timestamp', 'unknown'),
                'hours': hourly,
                'generated_at': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"LSTM 24h forecast error (falling back): {e}")

    # Fallback: direct Open-Meteo hourly forecast
    try:
        hourly = open_meteo_client.fetch_24h_hourly_forecast()
        return jsonify({
            'success': True,
            'source': 'open-meteo-fallback',
            'hours': hourly,
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 502


@app.route('/forecast/7day', methods=['GET'])
def forecast_7day():
    """Live 7-day forecast from Open-Meteo for University of Moratuwa (no ML model required)."""
    try:
        forecast = open_meteo_client.fetch_7day_forecast()
        return jsonify({'success': True, 'forecast': forecast})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 502

@app.route('/sensor-data/fallback', methods=['GET'])
def sensor_data_fallback():
    """Serve recent rows from the local CSV dataset when Firebase is unavailable."""
    limit = request.args.get('limit', default=48, type=int)
    records = load_csv_sensor_fallback(limit=limit)
    latest = records[-1] if records else None
    return jsonify({
        'success': True,
        'source': 'csv-fallback',
        'count': len(records),
        'latest': latest,
        'records': records,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make weather prediction
    
    Expected JSON payload:
    {
        "data": [
            {
                "datetime": "2024-01-01T00:00:00",
                "temperature": 25.0,
                "humidity": 65.0,
                ...
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json() or {}

        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400

        df = pd.DataFrame(data['data'])
        predictions = prediction_service.predict(df)

        return jsonify({
            'success': True,
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict-from-firebase', methods=['POST'])
def predict_from_firebase():
    """
    Fetch latest data from Firebase and make prediction.

    The service uses secure configuration from the environment or .env file.
    Optional request payload fields:
    {
        "firebase_url": "https://your-project.firebaseio.com"
    }
    """
    try:
        import firebase_admin
        from firebase_admin import db
        from firebase_admin import credentials

        data = request.get_json() or {}
        firebase_url = data.get('firebase_url') or config.FIREBASE_DATABASE_URL
        if not firebase_url:
            return jsonify({'error': 'Missing "firebase_url" field and no FIREBASE_DATABASE_URL configured'}), 400

        if not firebase_admin._apps:
            if config.FIREBASE_CREDENTIALS and os.path.exists(config.FIREBASE_CREDENTIALS):
                cred = credentials.Certificate(config.FIREBASE_CREDENTIALS)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': firebase_url
                })
            else:
                firebase_admin.initialize_app(options={
                    'databaseURL': firebase_url
                })

        ref = db.reference('/weather_data')
        cutoff_time = int((datetime.now() - timedelta(hours=24)).timestamp())
        query = ref.order_by_key().start_at(str(cutoff_time))
        recent_data = query.get()

        if not recent_data:
            return jsonify({'error': 'No recent data found in Firebase'}), 404

        records = []
        for key, reading in recent_data.items():
            if not isinstance(reading, dict):
                continue
            row = dict(reading)
            if str(key).isdigit():
                row['timestamp'] = int(key)
                row['datetime'] = datetime.fromtimestamp(int(key))
            if 'weather' in reading and isinstance(reading['weather'], dict):
                row.update(reading['weather'])
            if 'air_quality' in reading and isinstance(reading['air_quality'], dict):
                row.update(reading['air_quality'])
            records.append(row)

        if not records:
            return jsonify({'error': 'No valid weather records found in Firebase'}), 404

        df = pd.DataFrame(records)
        predictions = prediction_service.predict(df)
        return jsonify({
            'success': True,
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = config.API_PORT
    print("\n=== Weather Prediction API ===")
    print(f"Model loaded: {prediction_service.model is not None}")
    print(f"\nStarting server on http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
