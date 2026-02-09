FROM python:3.12-slim

ARG INSTALL_RL_DEPS=0

# Set working directory
WORKDIR /app

# System deps for OCR (Tesseract) + Chinese language pack
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		tesseract-ocr \
		tesseract-ocr-chi-tra \
	&& rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (better layer caching during development)
COPY requirements.txt requirements-rl.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
	&& if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi \
	&& if [ "$INSTALL_RL_DEPS" = "1" ] && [ -f requirements-rl.txt ]; then pip install --no-cache-dir -r requirements-rl.txt; fi

# Copy application files
COPY . /app

# Expose port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]