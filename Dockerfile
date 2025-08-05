FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OCR and Arabic text processing
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libtesseract-dev \
    # Image processing
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    # Arabic fonts
    fonts-arabeyes \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip config set global.timeout 300 && \
    pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["streamlit", "run", "app.py", "--server.maxUploadSize=200"]