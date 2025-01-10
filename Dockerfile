# Image de base
FROM python:3.10-slim

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    xvfb \  
    dbus-x11 \  
    && rm -rf /var/lib/apt/lists/*

# Définition des variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROME_BIN=/usr/bin/chromium \
    CHROME_DRIVER=/usr/bin/chromedriver

# Création et définition du répertoire de travail
WORKDIR /app

# Copie des requirements avant le reste du code
# (optimisation du cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code
COPY . .

# Création d'un utilisateur non-root pour la sécurité
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]