from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize — uses models/text-embedding-004 under the hood
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Embed a single piece of text
vector = embeddings.embed_query("What is machine learning?")

print(f"Type: {type(vector)}")          # list
print(f"Dimensions: {len(vector)}")     # 768
print(f"First 5 values: {vector[:5]}")  # [-0.023, 0.041, ...]

# Two ways to embed:
# embed_query()     → for questions / search queries (single string)
# embed_documents() → for your documents (list of strings)

doc_vectors = embeddings.embed_documents([
    "Python is a programming language",
    "LangChain is an AI framework",
    "RAG stands for Retrieval Augmented Generation"
])
print(f"Embedded {len(doc_vectors)} docs")  # 3
print(f"Each has {len(doc_vectors[0])} dims")  # 768