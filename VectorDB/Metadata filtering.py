from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Imagine you have docs from multiple sources
docs = [
    Document(page_content="Python tutorial page 1", metadata={"source": "python_tutorial", "chapter": 1}),
    Document(page_content="Python tutorial page 2", metadata={"source": "python_tutorial", "chapter": 2}),
    Document(page_content="Django docs chapter 1",  metadata={"source": "django_docs",     "chapter": 1}),
    Document(page_content="Django forms reference",  metadata={"source": "django_docs",     "chapter": 3}),
    Document(page_content="LangChain quickstart",    metadata={"source": "langchain_docs",  "chapter": 1}),
]

vs = Chroma.from_documents(docs, embeddings)

# Search ONLY in python_tutorial docs
results = vs.similarity_search(
    query="how to write a function",
    k=2,
    filter={"source": "python_tutorial"}  # Chroma filter syntax
)

# Search ONLY chapter 1s across all sources
results = vs.similarity_search(
    query="getting started",
    k=3,
    filter={"chapter": 1}
)

# Multi-condition filter
results = vs.similarity_search(
    query="forms and validation",
    k=2,
    filter={"$and": [
        {"source": {"$eq": "django_docs"}},
        {"chapter": {"$gte": 2}}
    ]}
)

# Retriever with filter built in
django_retriever = vs.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"source": "django_docs"}
    }
)