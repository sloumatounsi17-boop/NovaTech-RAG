# Image Python légère
FROM python:3.11-slim
# Dossier de travail dans le conteneur
WORKDIR /app
# Copier la liste des dépendances
COPY requirements.txt .
# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt
# Copier le code de l'application
COPY app.py .
# Copier les documents utilisés par le RAG
COPY documents ./documents
# Port utilisé par Gradio
EXPOSE 7860
# Lancer l'application
CMD ["python", "app.py"]