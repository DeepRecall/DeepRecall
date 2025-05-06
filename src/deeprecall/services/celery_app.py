import os
from celery import Celery

broker_url=os.getenv("BROKER_URL", "amqp://guest:guest@rabbitmq:30245"),

celery_app = Celery(
        "RAG",
        broker=broker_url,
    )