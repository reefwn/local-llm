from fastapi import FastAPI, Request
from api import rag
from api.env import FILES_PATH, MODEL_NAME, OLLAMA_API
from api.watcher import start_watcher
import requests
from fastapi.responses import StreamingResponse

app = FastAPI()
watcher = None

@app.on_event("startup")
def startup_event():
    global watcher
    warm_up_ollama()
    rag.build_index()
    watcher = start_watcher(FILES_PATH)

@app.on_event("shutdown")
def shutdown_event():
    if watcher:
        watcher.stop()
        watcher.join()


def warm_up_ollama():
    try:
        print("Warming up LLM...")
        requests.post(
            f"{OLLAMA_API}/api/generate",
            json={
                "model": MODEL_NAME, 
                "prompt": "Hello", 
                "stream": False
            },
            timeout=60
        )
    except Exception as e:
        print(f"Warmup failed: {e}")

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")

    return StreamingResponse(
        rag.query_index(question),
        media_type="text/event-stream"
    )
