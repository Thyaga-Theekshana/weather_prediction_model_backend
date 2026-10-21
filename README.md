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

5. Run the prediction API:

```bash
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
