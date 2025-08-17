# 🚀 AI Internship Projects - Render Deployment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install essential system dependencies for Render
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-render.txt ./requirements.txt

# Install Python dependencies optimized for Render
RUN pip install --no-cache-dir --timeout=1000 --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Install spaCy language model
RUN python -m spacy download en_core_web_sm || echo "spaCy model will download at runtime"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/Chatbot_AI/vectorstore \
    && mkdir -p /app/uploads \
    && mkdir -p /app/temp

# Create Render-optimized startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting AI Internship Dashboard..."\n\
\n\
# Download spaCy model if needed\n\
echo "Checking spaCy model..."\n\
python -c "import spacy; spacy.load(\"en_core_web_sm\")" 2>/dev/null || {\n\
  echo "Downloading spaCy model..."\n\
  python -m spacy download en_core_web_sm\n\
}\n\
\n\
# Get port from Render environment or default to 8501\n\
PORT=${PORT:-8501}\n\
echo "Using port: $PORT"\n\
\n\
# Start Streamlit with Render configuration\n\
echo "Starting Streamlit server..."\n\
exec streamlit run master_app_enterprise.py \\\n\
  --server.port=$PORT \\\n\
  --server.address=0.0.0.0 \\\n\
  --server.headless=true \\\n\
  --server.enableCORS=false \\\n\
  --server.enableXsrfProtection=false \\\n\
  --server.maxUploadSize=200 \\\n\
  --server.maxMessageSize=200 \\\n\
  --server.enableWebsocketCompression=true \\\n\
  --browser.gatherUsageStats=false\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port (Render will override this)
EXPOSE 8501

# Set environment variables for Render
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
ENV STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=true


# Run the application
CMD ["/app/start.sh"]