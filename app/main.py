from fastapi import FastAPI, Request
from app import rag
from app.env import FILES_PATH
from app.watcher import start_watcher

app = FastAPI()
watcher = None

@app.on_event("startup")
def startup_event():
    global watcher
    rag.build_index()
    watcher = start_watcher(FILES_PATH)

@app.on_event("shutdown")
def shutdown_event():
    if watcher:
        watcher.stop()
        watcher.join()

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    answer = rag.query_index(question)

    return {"answer": answer}
