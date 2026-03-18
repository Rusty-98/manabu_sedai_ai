from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def cosine_similarity(v1: list, v2: list) -> float:
    """Higher = more similar. Range: -1 to 1."""
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Embed several sentences
sentences = [
    "I love playing cricket",         # query
    "Cricket is my favourite sport",  # very similar
    "Football is a popular game",     # somewhat related (sports)
    "Python is a programming language", # unrelated
    "I enjoy watching IPL matches",   # similar to query
]

vectors = embeddings.embed_documents(sentences)
query_vec = vectors[0]  # "I love playing cricket"

print(f"Query: '{sentences[0]}'\n")
for i in range(1, len(sentences)):
    score = cosine_similarity(query_vec, vectors[i])
    bar = "█" * int(score * 30)
    print(f"{score:.3f} {bar}")
    print(f"       '{sentences[i]}'\n")

# Output (approximately):
# 0.921 ██████████████████████████   "Cricket is my favourite sport"
# 0.743 ██████████████████████        "Football is a popular game"
# 0.412 ████████████                  "Python is a programming language"
# 0.887 ██████████████████████████   "I enjoy watching IPL matches"