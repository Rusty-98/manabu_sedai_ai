from dotenv import load_dotenv
load_dotenv()

import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Sample documents (in real RAG, these are your chunks)
documents = [
    "LangChain is a framework for building LLM-powered applications.",
    "RAG combines retrieval with generation for accurate answers.",
    "Gemini is Google's multimodal AI model family.",
    "Vector databases store embeddings for semantic search.",
    "Chunking splits documents into smaller pieces for embedding.",
    "Cosine similarity measures how close two vectors are.",
    "Python is widely used for AI and ML development.",
    "FastAPI is a modern Python web framework for building APIs.",
    "Pydantic provides data validation using Python type annotations.",
    "ChromaDB is an open-source vector database for AI applications.",
]

# embed_documents handles batching automatically
start = time.time()
vectors = embeddings.embed_documents(documents)
elapsed = time.time() - start

print(f"Embedded {len(vectors)} docs in {elapsed:.2f}s")
print(f"Shape: {len(vectors)} x {len(vectors[0])}")  # 10 x 768

# Store with metadata for later retrieval
doc_store = [
    {"text": doc, "vector": vec, "id": i}
    for i, (doc, vec) in enumerate(zip(documents, vectors))
]

# Now you can search:
def search(query: str, top_k: int = 3):
    import numpy as np
    q_vec = np.array(embeddings.embed_query(query))
    scores = []
    for item in doc_store:
        v = np.array(item["vector"])
        score = float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v)))
        scores.append((score, item["text"]))
    scores.sort(reverse=True)
    return scores[:top_k]

results = search("How do I store vectors?")
for score, text in results:
    print(f"{score:.3f} | {text}")