# 🧠 Local LLM System

This project is a fully Dockerized **Retrieval-Augmented Generation (RAG)** system that lets you ask questions over documents in a folder — completely offline.

It features:
- 🤖 **Ollama** for local LLM inference (e.g. Mistral, Phi-3)
- 🔍 **LlamaIndex** for document indexing and retrieval
- 🗂️ **ChromaDB** for fast vector storage
- ⚡ **FastAPI** backend with streaming support (SSE)
- 🌐 **Streamlit** frontend UI with live token streaming
- 🔄 **Live reload & folder watching** for auto reindex

---

## 📦 Features

- ✅ Document indexing from `./files` folder
- ✅ LLM query answering with sources
- ✅ Streamed response via Server-Sent Events (SSE)
- ✅ Hot-reload dev mode (`--reload` + volume mounts)
- ✅ Configurable LLM and embedding models via `.env`

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/reefwn/local-llm
cd local-llm
```

### 2. Pull required Ollama model

```bash
ollama pull mistral
```

### 3. Start all services
```bash
docker-compose up -d
```

### 4. Open your browser
- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## ⚙️ Environment Configuration

Create a .env file

```dotenv
MODEL_NAME=mistral
EMBEDDING_MODEL=intfloat/e5-small-v2
FILES_PATH=/app/files
OLLAMA_API=http://ollama:11434
CHROMA_HOST=chroma
CHROMA_PORT=8000
```

## 🗂️ Folder Structure

```bash
.
├── api/               # FastAPI backend + LlamaIndex logic
├── ui/                # Streamlit frontend
├── files/             # Drop your documents here (PDF, TXT, MD)
├── Dockerfile.api     # API Dockerfile
├── Dockerfile.ui      # UI Dockerfile
├── docker-compose.yml
├── requirements.api.txt
├── requirements.ui.txt
└── .env
```

## 🧪 Supported Models

Any [Ollama-supported](https://ollama.com/library) model like:
  - mistral, mistral:instruct, mistral:Q4_0
  - phi3:mini:Q4_0
  - gemma:2b, tinyllama:chat
  - llama2:7b, etc.

Make sure to ollama pull <model> before using it.

## ❓ Example Query

> "Summarize all documents in simple terms."

Or:

> "What are the company leave policies mentioned in the PDFs?"

## ❤️ Credits

Built with:
- [Ollama](https://ollama.com)
- [LlamaIndex](https://www.llamaindex.ai)
- [ChromaDB](https://www.trychroma.com)
- [Streamlit](https://streamlit.io)
- [FastAPI](https://fastapi.tiangolo.com)

Files:
- [Medicine_Details.csv](https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details)
- [who-model-formulary-2008.pdf](https://iris.who.int/handle/10665/44053)
