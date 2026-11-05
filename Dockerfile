FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure required directories exist
RUN mkdir -p models data logs

# Back4App uses port 8000 by default
EXPOSE 8000

# Start the Flask WSGI app through its ASGI adapter with Uvicorn
CMD ["uvicorn", "app:asgi_app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
