from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")  # or "models/text-embedding-004" for older
PERSIST_DIR = "./chroma_db"  # folder to store the DB

# ─── FIRST RUN: Create and save ───────────────────────────
if not os.path.exists(PERSIST_DIR):
    print("Building vector store...")
    docs = [
        Document(page_content="...", metadata={"source": "doc1.txt"}),
        # ... your documents
    ]
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,  # saves to disk
        collection_name="my_rag_db"
    )
    print(f"Saved {vectorstore._collection.count()} vectors")

# ─── SUBSEQUENT RUNS: Load existing ───────────────────────
else:
    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="my_rag_db"
    )
    print(f"Loaded {vectorstore._collection.count()} vectors")

# ─── Add new documents later ──────────────────────────────
new_doc = Document(
    page_content="New information added after initial indexing.",
    metadata={"source": "doc2.txt", "added": "2025-01"}
)
vectorstore.add_documents([new_doc])

# ─── Delete by metadata ───────────────────────────────────
vectorstore.delete(
    where={"source": "doc1.txt"}  # remove all from doc1.txt
)