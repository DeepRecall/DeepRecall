from fastapi import FastAPI
from deeprecall.services.celery_app import celery_app
from deeprecall.rag.embed.task import create_rag

app = FastAPI()


@app.post("/populate_rag")
async def start_processing(data: str):
    task = create_rag.delay()
    return {"task_id": task.id}


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task = celery_app.AsyncResult(task_id)
    return {"status": task.status, "result": task.result}
