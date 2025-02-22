# Image de base avec CUDA
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Configurer l'environnement Python
ENV DEBIAN_FRONTEND=noninteractive

# Installer Python et les dépendances système
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Définir Python 3.10 comme version par défaut
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY api.py bench_config.py init_environment.py ./
COPY entrypoint.sh .

# Installer les dépendances Python
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Créer les dossiers nécessaires
RUN mkdir -p models data evaluation_results

# Exposer le port
EXPOSE 8000

# Rendre le script d'entrée exécutable
RUN chmod +x entrypoint.sh

# Commande par défaut
ENTRYPOINT ["./entrypoint.sh"] 