from dotenv import load_dotenv
load_dotenv()
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Instead of splitting by size, splits when topic CHANGES
# Uses embeddings to detect meaning shifts between sentences
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # split at biggest jumps
    breakpoint_threshold_amount=95           # top 5% of jumps = new chunk
)

text = """
Python is a high-level programming language known for simplicity.
It was created by Guido van Rossum in 1991.
Python supports multiple programming paradigms.

Machine learning is a subset of artificial intelligence.
ML algorithms learn from data to make predictions.
Deep learning uses neural networks with many layers.

Football is played between two teams of eleven players.
The FIFA World Cup is held every four years.
Brazil has won the most World Cup titles.
"""

chunks = splitter.split_text(text)

# Should produce 3 chunks — Python, ML, Football
# Because those are 3 distinct topics
print(f"Chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(c[:150])