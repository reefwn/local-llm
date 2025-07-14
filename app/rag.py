from app.env import CHROMA_COLLECTION, EMBEDDING_MODEL, FILES_PATH, CHROMA_HOST, CHROMA_PORT, MODEL_NAME, OLLAMA_API
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb


client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_or_create_collection(CHROMA_COLLECTION)
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = None


def build_index():
    global index
    docs = SimpleDirectoryReader(FILES_PATH).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )


def query_index(question: str):
    global index
    if not index:
        build_index()

    llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_API,
        request_timeout=300
    )

    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="compact",
        return_source=True
    )

    response = query_engine.query(question)

    sources = [
        {
            "text": node.text,
            "metadata": node.metadata
        }
        for node in response.source_nodes
    ]

    return {
        "answer": str(response),
        "sources": sources
    }
