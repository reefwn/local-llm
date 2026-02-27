"""
LangGraph RAG pipeline: retrieve → grade → generate (with fallback).
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from api.env import MODEL_NAME, OLLAMA_API
from api import rag
import json

SYSTEM_PROMPT = """You are a certified pharmacist assistant.

Only suggest medications that are explicitly found in the provided context. Do not invent any medication names or dosages.

If a symptom does not match a known medication in the context, say:
"I couldn't find a recommended medicine for this condition. Please consult a pharmacist."

For each recommendation, provide:
- The medicine name
- Its use case
- Dosage (if known)
- Side effects and warnings"""

GRADER_PROMPT = """You are a relevance grader. Given a user question and retrieved documents, determine if the documents contain information relevant to answering the question.

Reply with ONLY "relevant" or "not_relevant"."""


# --- State ---

class GraphState(TypedDict):
    question: str
    documents: list
    generation: str
    sources: list


# --- Nodes ---

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents from the vector store."""
    question = state["question"]
    retriever = rag.get_retriever()
    nodes = retriever.retrieve(question)

    documents = []
    sources = []
    for node in nodes:
        documents.append(node.text)
        sources.append({"text": node.text, "metadata": node.metadata})

    return {"documents": documents, "sources": sources}


def grade_documents(state: GraphState) -> GraphState:
    """Grade retrieved documents for relevance."""
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_API, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_PROMPT),
        ("human", "Question: {question}\n\nDocuments:\n{documents}"),
    ])
    chain = prompt | llm

    docs_text = "\n---\n".join(state["documents"]) if state["documents"] else "(no documents)"
    result = chain.invoke({"question": state["question"], "documents": docs_text})

    grade = result.content.strip().lower()
    if "not_relevant" in grade:
        return {"documents": []}
    return state


def generate(state: GraphState) -> GraphState:
    """Generate answer using retrieved context."""
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_API, temperature=0.3)
    context = "\n---\n".join(state["documents"]) if state["documents"] else "No relevant documents found."

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": state["question"]})
    return {"generation": result.content}


def fallback(state: GraphState) -> GraphState:
    """Fallback when no relevant documents are found."""
    return {
        "generation": "I couldn't find a recommended medicine for this condition. Please consult a pharmacist.",
        "sources": [],
    }


# --- Conditional edge ---

def decide_after_grading(state: GraphState) -> str:
    """Route to generate or fallback based on grading result."""
    if not state.get("documents"):
        return "fallback"
    return "generate"


# --- Build graph ---

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# Compiled graph (singleton)
rag_graph = None


def get_graph():
    global rag_graph
    if rag_graph is None:
        rag_graph = build_graph()
    return rag_graph


def run_query(question: str):
    """Run the RAG graph and yield SSE tokens."""
    graph = get_graph()
    result = graph.invoke({"question": question, "documents": [], "generation": "", "sources": []})

    # Stream the generation as SSE
    for char in result["generation"]:
        yield f"data: {char}\n\n"

    # Send sources
    source_payload = result.get("sources", [])
    yield f"event: sources\ndata: {json.dumps(source_payload)}\n\n"
