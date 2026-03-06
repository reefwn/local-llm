from api.env import EMBEDDING_MODEL, FILES_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
import os
import torch

FAISS_PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "./faiss_index")
FAISS_DIM = int(os.getenv("FAISS_DIM", "384"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL, 
    device=device,
    embed_batch_size=EMBED_BATCH_SIZE
)
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
index = None


def get_file_metadata(file_path: str):
    stat = os.stat(file_path)
    return {
        "file_path": file_path,
        "last_modified": str(int(stat.st_mtime))
    }


def _load_persisted_index():
    """Load existing FAISS index from disk if available."""
    if os.path.exists(os.path.join(FAISS_PERSIST_DIR, "default__vector_store.json")):
        vector_store = FaissVectorStore.from_persist_dir(FAISS_PERSIST_DIR)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=FAISS_PERSIST_DIR)
        return load_index_from_storage(storage_context, embed_model=embed_model)
    return None


def indexing(docs):
    faiss_index = faiss.IndexFlatL2(FAISS_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    idx = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
        transformations=[splitter]
    )
    idx.storage_context.persist(persist_dir=FAISS_PERSIST_DIR)
    return idx


def build_index():
    global index

    # Try loading persisted index
    index = _load_persisted_index()

    # Load all files
    docs = []
    for file in os.listdir(FILES_PATH):
        file_path = os.path.join(FILES_PATH, file)
        if not os.path.isfile(file_path):
            continue

        print(f"[Indexing] Adding: {file}")
        loader = SimpleDirectoryReader(input_files=[file_path])
        file_docs = loader.load_data()

        meta = get_file_metadata(file_path)
        for doc in file_docs:
            doc.metadata.update(meta)
            docs.append(doc)

    if docs:
        index = indexing(docs)
        print(f"[Indexing] Indexed {len(docs)} docs with FAISS.")
    elif not index:
        print("[Indexing] No documents found. Index remains uninitialized.")


def get_retriever(similarity_top_k: int = 3):
    """Return a retriever from the current index for use by LangGraph."""
    global index
    if not index:
        build_index()
        if not index:
            raise RuntimeError("No documents available for retrieval. Please add documents to the FILES_PATH directory.")
    return index.as_retriever(similarity_top_k=similarity_top_k)
