FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies (use PyTorch CPU-only index to get lighter build)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Ensure required directories exist
RUN mkdir -p models data logs

# Back4App uses port 8000 by default
EXPOSE 8000

# Start Flask app with Gunicorn
CMD ["gunicorn", "prediction_api:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--workers", "2"]
