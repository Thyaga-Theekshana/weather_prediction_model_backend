import os
import shutil
import subprocess
import sys

if __name__ == "__main__":
    # Ensure required directories exist (Git doesn't track empty directories)
    for directory in ["models", "data", "logs"]:
        os.makedirs(directory, exist_ok=True)

    # Read Firebase credentials from environment secret and write to file
    firebase_json_str = os.environ.get("FIREBASE_JSON_SECRET")
    if firebase_json_str:
        with open("firebase_credentials.json", "w") as f:
            f.write(firebase_json_str)
        os.environ["FIREBASE_CREDENTIALS"] = "firebase_credentials.json"

    print("Starting Flask API...")
    gunicorn_path = shutil.which("gunicorn") or shutil.which("gunicorn.exe")
    if gunicorn_path:
        cmd = [
            sys.executable,
            "-m",
            "gunicorn",
            "prediction_api:app",
            "--bind",
            "0.0.0.0:8000",
            "--timeout",
            "120",
            "--workers",
            "2",
        ]
        subprocess.run(cmd, check=True)
    else:
        import prediction_api
        print("Gunicorn not found. Falling back to Flask development server on port 8000.")
        prediction_api.app.run(host="0.0.0.0", port=8000, debug=False)
