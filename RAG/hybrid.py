from dotenv import load_dotenv
load_dotenv()

# pip install rank-bm25 langchain-community

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

docs = [
    Document(page_content="Python 3.12 released with 5% performance boost.", metadata={"source": "news"}),
    Document(page_content="How to install Python on Ubuntu Linux.", metadata={"source": "tutorial"}),
    Document(page_content="Python vs JavaScript for backend development.", metadata={"source": "blog"}),
    Document(page_content="FastAPI is a Python web framework for APIs.", metadata={"source": "docs"}),
    Document(page_content="Django is a batteries-included Python framework.", metadata={"source": "docs"}),
]

# Vector retriever — finds semantically similar docs
vectorstore = Chroma.from_documents(docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# BM25 retriever — finds exact keyword matches (like Google's algorithm)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

# Ensemble — combines both with weights
# 0.6 semantic + 0.4 keyword = good balance
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

# Compare results
query = "Python web framework installation"
print("=== Vector only ===")
for d in vector_retriever.invoke(query):
    print(f"  {d.page_content[:60]}")

print("\n=== BM25 only ===")
for d in bm25_retriever.invoke(query):
    print(f"  {d.page_content[:60]}")

print("\n=== Hybrid (better) ===")
for d in ensemble_retriever.invoke(query):
    print(f"  {d.page_content[:60]}")