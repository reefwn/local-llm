import os
from dotenv import load_dotenv

load_dotenv()

FILES_PATH = os.getenv("FILES_PATH", "./files")
EMBEDDING_MODEL= os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
OLLAMA_API = os.getenv("OLLAMA_API", "http://ollama:11434")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
