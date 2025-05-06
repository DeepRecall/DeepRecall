from deeprecall.services.celery_app import celery_app

@celery_app.task
def process_data(data):
    # Simulate long-running task
    return f"Processed: {data}"