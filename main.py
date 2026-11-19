
import os
import json
import hashlib

from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END, MessagesState


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DATA_FILE = BASE_DIR / "data" / "scraped_data.json"

CHROMA_PATH = Path(
    os.getenv("CHROMA_DB_PATH", "chroma_db")
)

if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = BASE_DIR / CHROMA_PATH

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "real_estate_properties"
)

TOP_K = int(os.getenv("TOP_K", "6"))


# --------------------------------------------------
# 2. API CONFIGURATION
# --------------------------------------------------

def get_embeddings():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY missing in .env"
        )

    return OpenAIEmbeddings(
        model=os.getenv(
            "EMBEDDING_MODEL",
            "openai/text-embedding-3-small"
        ),
        api_key=api_key,
        base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1"
        )
    )


def get_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY missing in .env"
        )

    return ChatOpenAI(
        model=os.getenv(
            "LLM_MODEL",
            "openai/gpt-4o-mini"
        ),
        api_key=api_key,
        base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1"
        ),
        temperature=0
    )


# --------------------------------------------------
# 3. LOAD JSON DATA
# --------------------------------------------------

def load_json_documents():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"JSON file not found: {DATA_FILE}"
        )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Expected JSON to contain a list of records."
        )

    documents = []

    for record in records:

        if not isinstance(record, dict):
            continue

        content = str(record.get("content") or "").strip()

        if not content:
            continue

        metadata = {
            "source": str(record.get("source", "Unknown")),
            "title": str(record.get("title", "Untitled")),
            "url": str(record.get("url", "")),
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    print(f"Loaded {len(documents)} documents.")

    return documents


# --------------------------------------------------
# 4. CREATE VECTOR STORE
# --------------------------------------------------

def get_vector_store():

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PATH)
    )


# --------------------------------------------------
# 5. INGEST JSON INTO CHROMADB
# --------------------------------------------------

def ingest_data():

    print("\nLoading JSON data...")

    documents = load_json_documents()

    if not documents:
        print("No documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Rebuild the collection to prevent old or
    # duplicate chunks after a new ingestion.
    if CHROMA_PATH.exists():
        old_store = get_vector_store()
        old_store.delete_collection()

    vector_store = get_vector_store()

    ids = []

    for index, chunk in enumerate(chunks):

        identity = (
            chunk.metadata.get("url", "")
            + chunk.page_content
            + str(index)
        )

        ids.append(
            hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()
        )

        chunk.metadata["chunk_index"] = index

    batch_size = 100

    for start in range(0, len(chunks), batch_size):

        end = start + batch_size

        vector_store.add_documents(
            documents=chunks[start:end],
            ids=ids[start:end]
        )

        print(
            f"Stored {min(end, len(chunks))}"
            f"/{len(chunks)} chunks."
        )

    print("\nIngestion completed!")
    print(f"Original documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector database: {CHROMA_PATH}")


# --------------------------------------------------
# 6. LANGGRAPH STATE
# --------------------------------------------------

class RAGState(MessagesState):
    documents: list[Document]


# --------------------------------------------------
# 7. RETRIEVAL NODE
# --------------------------------------------------

def retrieve_node(state: RAGState):

    question = ""

    for message in reversed(state["messages"]):

        if isinstance(message, HumanMessage):
            question = str(message.content)
            break

    if not question:
        return {"documents": []}

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=TOP_K
    )

    print(
        f"Retrieved {len(documents)} chunks."
    )

    return {
        "documents": documents
    }


# --------------------------------------------------
# 8. FORMAT RETRIEVED CONTEXT
# --------------------------------------------------

def format_context(documents):

    if not documents:
        return "No relevant documents were retrieved."

    context = []

    for index, document in enumerate(documents, start=1):

        metadata = document.metadata

        context.append(
            f"""
DOCUMENT {index}

Website: {metadata.get("source", "Unknown")}
Title: {metadata.get("title", "Untitled")}
URL: {metadata.get("url", "Unknown")}

Content:
{document.page_content}
"""
        )

    return "\n\n".join(context)


# --------------------------------------------------
# 9. GENERATION NODE
# --------------------------------------------------

def generate_node(state: RAGState):

    llm = get_llm()

    context = format_context(
        state.get("documents", [])
    )

    system_prompt = f"""
You are RealEstateAI, a real estate information
assistant.

You answer questions using the supplied context
retrieved from DarGlobal and Wasalt.

RULES:

1. Use only the retrieved context.
2. Never invent property names, prices, locations,
   sizes, amenities, or availability.
3. If information is missing, say that it was not
   found in the available data.
4. If the user specifies a city, only describe
   properties whose location is supported by context.
5. Include the source website and URL for factual
   property information.
6. Do not claim a property is currently available
   unless the context confirms it.
7. Keep answers clear and easy to understand.

RETRIEVED CONTEXT:

{context}
"""

    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# --------------------------------------------------
# 10. BUILD LANGGRAPH
# --------------------------------------------------

builder = StateGraph(RAGState)

builder.add_node(
    "retrieve",
    retrieve_node
)

builder.add_node(
    "generate",
    generate_node
)

builder.add_edge(
    START,
    "retrieve"
)

builder.add_edge(
    "retrieve",
    "generate"
)

builder.add_edge(
    "generate",
    END
)

rag_graph = builder.compile()


# --------------------------------------------------
# 11. CHAT FUNCTION
# --------------------------------------------------

def ask_agent(question, chat_history=None):

    if not question.strip():
        return "Please enter a question."

    messages = []

    for item in (chat_history or []):

        if isinstance(item, BaseMessage):
            messages.append(item)

        elif isinstance(item, dict):

            role = item.get("role")
            content = item.get("content", "")

            if role == "user":
                messages.append(
                    HumanMessage(content=content)
                )

            elif role == "assistant":
                messages.append(
                    AIMessage(content=content)
                )

    messages.append(
        HumanMessage(content=question)
    )

    result = rag_graph.invoke({
        "messages": messages
    })

    answer = result["messages"][-1].content

    if isinstance(answer, list):
        answer = "\n".join(
            str(item.get("text", item))
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    return str(answer)


# --------------------------------------------------
# 12. COMMAND LINE
# --------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ingest",
        action="store_true"
    )

    args = parser.parse_args()

    if args.ingest:
        ingest_data()
    else:
        print(
            "Use: python -m app.main --ingest"
        )
