from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')


def get_env(name, default=None, required=False):
    value = os.getenv(name, default)
    if required and (value is None or value == ''):
        raise EnvironmentError(f'Missing required environment variable: {name}')
    return value


MODEL_DIR = get_env('MODEL_DIR', 'models')
FIREBASE_DATABASE_URL = get_env('FIREBASE_DATABASE_URL', get_env('FIREBASE_URL'))
FIREBASE_CREDENTIALS = get_env('FIREBASE_CREDENTIALS', get_env('GOOGLE_APPLICATION_CREDENTIALS'))

if FIREBASE_CREDENTIALS:
    FIREBASE_CREDENTIALS = str(Path(FIREBASE_CREDENTIALS).expanduser())


def require_firebase_configuration():
    if not FIREBASE_DATABASE_URL:
        raise EnvironmentError(
            'FIREBASE_DATABASE_URL is required. Set it in .env or your environment.'
        )
    if not FIREBASE_CREDENTIALS:
        raise EnvironmentError(
            'FIREBASE_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS is required. '
            'Set a secure path to your Firebase service account JSON file in .env or the environment.'
        )
    return FIREBASE_DATABASE_URL, FIREBASE_CREDENTIALS
