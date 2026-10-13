FROM python:3.11-slim

WORKDIR /app

# Dépendances système requises pour OpenCV et le traitement d'images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source et de l'interface
COPY src/ /app/src/
COPY api/ /app/api/
COPY web/ /app/web/
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# Exposer le port de FastAPI
EXPOSE 8000

# Lancement du serveur FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
