from api.env import CHROMA_COLLECTION, EMBEDDING_MODEL, FILES_PATH, CHROMA_HOST, CHROMA_PORT, MODEL_NAME, OLLAMA_API, CHUNK_SIZE, CHUNK_OVERLAP
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
import time


client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_or_create_collection(CHROMA_COLLECTION)
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = None


def build_index():
    global index
    docs = SimpleDirectoryReader(
        input_dir=FILES_PATH,
        recursive=True,
    ).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
        transformations=[splitter]
    )


def query_index(question: str):
    global index
    if not index:
        build_index()

    llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_API,
        request_timeout=300,
        stream=True,
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=2,
        return_source=False,
        streaming=True,
    )
    response = query_engine.query(question)

    for token in response.response_gen:
        yield f"data: {token}\n\n"