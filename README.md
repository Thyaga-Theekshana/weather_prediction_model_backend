---
title: Weather Backend
emoji: 🌤️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---
# Weather Prediction Model Backend

This repository contains the backend model and prediction pipeline for the weather and air quality forecasting system.

## Contents

- `train_lstm_model.py` - LSTM training and model persistence
- `train_xgboost_model.py` - XGBoost baseline training and prediction
- `prediction_api.py` - Flask API for serving predictions
- `live_prediction_service.py` - Firebase-backed live prediction service
- `auto_retrain.py` - Automated retraining pipeline
- `firebase_export.py` - Firebase data export utilities
- `generate_sample_data.py` - Sample data generation helper
- `moratuwa_sensors_20k_dataset.csv` - Example dataset for training
- `requirements.txt` - Backend dependency list

## Usage

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Create your environment file:

```bash
copy .env.example .env
```

3. Update `.env` with your Firebase project settings and credentials path.

4. Train the model:

```bash
python train_lstm_model.py
```

5. Run the prediction API on the port used by the frontend:

```bash
python app.py
```

The API is available at `http://localhost:8000` and provides a 24-hour hourly
forecast at `GET http://localhost:8000/forecast/24h`. `app.py` adapts Flask's
WSGI application for Uvicorn, so the same startup command works on Windows and
Linux without Gunicorn.

For a direct Flask run, use the same port explicitly:

```bash
API_PORT=8000 python prediction_api.py
```

On Windows PowerShell:

```powershell
$env:API_PORT = "8000"
python prediction_api.py
```

6. Run the live prediction service:

```bash
python live_prediction_service.py
```

## Notes

- This repository is intentionally separate from the frontend project.
- Model artifacts are stored in `models/` and are excluded from version control.
- Secrets and service account files must never be checked in.
- Use `.env` and environment variables to keep credentials out of source control.
- See `.env.example` for the required configuration keys.
- `generate_sample_data.py` creates clearly synthetic, smooth Moratuwa demo readings
  bounded to 27-32 C, 75-85% humidity, 1010-1013 hPa, 2-5.5 m/s wind,
  8-25 ug/m3 PM2.5, 20-50 ug/m3 PM10, and AQI 35-65. It writes the CSV to
  the current working directory; use a temporary directory when you do not
  intend to replace the bundled example dataset. The API applies those bounds
  only to its local CSV fallback, not to live sensor readings.
