# In backend/app/worker.py

from celery import Celery

# URL del tuo broker Redis (avviato con Docker)
REDIS_URL = "redis://localhost:6379/0"

# Crea l'istanza dell'app Celery
# 'include' dice a Celery di cercare i task nel file 'app.tasks'
celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL, # Usiamo Redis anche per i risultati
    include=['app.tasks'] # Punta al file che creeremo al prossimo step
)

# Configurazioni opzionali
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Rome',
)