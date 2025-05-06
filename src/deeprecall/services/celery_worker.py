# worker_main.py
from celery.bin.worker import worker
from deeprecall.services.celery_app import celery_app

# Initialize worker
celery_worker = worker(app=celery_app)

# Start worker with custom options (e.g., default -l info -c 1)
options = {
    "loglevel": "INFO",
    "concurrency": 1,  # containers are parallel workers
}

# Start Celery worker as subprocess
celery_worker.run(**options)