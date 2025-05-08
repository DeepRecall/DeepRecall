from fastapi import FastAPI
from deeprecall.services.celery_app import celery_app
from deeprecall.rag.embed.task import process_data

app = FastAPI()

@app.post("/process")
async def start_processing(data: str):
    task = process_data.delay(data)
    return {"task_id": task.id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task = celery_app.AsyncResult(task_id)
    return {"status": task.status, "result": task.result}