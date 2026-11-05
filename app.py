import os

from a2wsgi import WSGIMiddleware


def create_asgi_app():
    for directory in ("models", "data", "logs"):
        os.makedirs(directory, exist_ok=True)

    firebase_json_str = os.environ.get("FIREBASE_JSON_SECRET")
    if firebase_json_str:
        with open("firebase_credentials.json", "w", encoding="utf-8") as credentials_file:
            credentials_file.write(firebase_json_str)
        os.environ["FIREBASE_CREDENTIALS"] = "firebase_credentials.json"

    import prediction_api

    return WSGIMiddleware(prediction_api.app)


asgi_app = create_asgi_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", os.environ.get("API_PORT", "8000")))
    print(f"Starting prediction API with Uvicorn on port {port}...")
    uvicorn.run(asgi_app, host="0.0.0.0", port=port, timeout_keep_alive=120)
