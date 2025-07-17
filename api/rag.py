from api.env import CHROMA_COLLECTION, EMBEDDING_MODEL, FILES_PATH, CHROMA_HOST, CHROMA_PORT, MODEL_NAME, OLLAMA_API, CHUNK_SIZE, CHUNK_OVERLAP
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
import os
import json

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_or_create_collection(CHROMA_COLLECTION)
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
vector_store = ChromaVectorStore(chroma_collection=collection)
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
index = None


def get_file_metadata(file_path: str):
    stat = os.stat(file_path)
    return {
        "file_path": file_path,
        "last_modified": str(int(stat.st_mtime))
    }


def indexing(docs):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
        transformations=[splitter]
    )

def build_index():
    global index

    # Get or create Chroma collection
    existing_metadatas = collection.get(include=["metadatas"])["metadatas"]

    # Build a set of (file_path, last_modified) for deduplication
    existing_signatures = set()
    for meta in existing_metadatas:
        key = (meta.get("file_path"), meta.get("last_modified"))
        existing_signatures.add(key)

    print(f"[Indexing] Found {len(existing_signatures)} files in Chroma.")

    # Load new/changed files
    docs = []
    for file in os.listdir(FILES_PATH):
        file_path = os.path.join(FILES_PATH, file)
        if not os.path.isfile(file_path):
            continue

        meta = get_file_metadata(file_path)
        signature = (meta["file_path"], meta["last_modified"])
        if signature in existing_signatures:
            print(f"[Indexing] Skipping already indexed: {file}")
            continue

        print(f"[Indexing] Adding: {file}")
        loader = SimpleDirectoryReader(input_files=[file_path])
        file_docs = loader.load_data()

        # Add metadata to each doc
        for doc in file_docs:
            doc.metadata.update(meta)
            docs.append(doc)

    if docs:
        index = indexing(docs)
        print(f"[Indexing] Added {len(docs)} new docs to Chroma.")
    else:
        print("[Indexing] No new documents to index.")

    # Build index for retrieval
    index = indexing([])


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
        return_source=True,
        streaming=True,
    )
    response = query_engine.query(question)

    for token in response.response_gen:
        yield f"data: {token}\n\n"

    # ✅ Send final source metadata as JSON
    source_payload = [
        {
            "text": node.text,
            "metadata": node.metadata
        }
        for node in response.source_nodes
    ]
    yield f"event: sources\ndata: {json.dumps(source_payload)}\n\n"