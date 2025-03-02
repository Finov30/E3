FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers requis
COPY requirements.txt .
COPY . .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Création des dossiers nécessaires
RUN mkdir -p /app/mlruns \
    /app/mlflow_registry \
    /app/mlflow_artifacts \
    /app/saved_models \
    /app/logs \
    /app/confusion_analysis \
    /app/benchmark_results \
    && chmod -R 777 /app

# Exposition du port
EXPOSE 8000
EXPOSE 5000

# Commande de démarrage
CMD ["./entrypoint.sh"] 