from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Your documents
docs = [
    Document(page_content="Python is great for data science and AI.", metadata={"topic": "python"}),
    Document(page_content="LangChain simplifies building LLM applications.", metadata={"topic": "langchain"}),
    Document(page_content="RAG retrieves relevant docs before generating answers.", metadata={"topic": "rag"}),
    Document(page_content="Chroma is a vector database for AI apps.", metadata={"topic": "vectordb"}),
    Document(page_content="Gemini is Google's flagship AI model.", metadata={"topic": "gemini"}),
]

# Create vector store — embeds all docs automatically
# (calls Gemini embeddings API internally)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="my_collection"
)

# Similarity search — find 3 most relevant docs
results = vectorstore.similarity_search(
    query="How do I build an AI application?",
    k=3
)
for doc in results:
    print(f"[{doc.metadata['topic']}] {doc.page_content}")

# With scores (lower distance = more similar in Chroma)
results_with_scores = vectorstore.similarity_search_with_score(
    "vector database for storing embeddings", k=3
)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:60]}")