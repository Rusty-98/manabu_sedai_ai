from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
LangChain is a framework for developing applications powered by large language models.
It provides tools for chaining LLM calls, managing prompts, and building agents.

RAG (Retrieval Augmented Generation) is a technique that combines information retrieval
with text generation. Instead of relying solely on the model's training data, RAG
fetches relevant documents and includes them in the prompt context.

Vector databases store embeddings — numerical representations of text. When you
search, your query is also embedded and compared against stored vectors using
cosine similarity to find the most relevant results.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # max chars per chunk
    chunk_overlap=40,     # overlap between chunks (~20%)
    length_function=len,  # how to measure size
    separators=["\n\n", "\n", ". ", " ", ""]  # tries each in order
)

chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
    print()

# With Documents (LangChain's unit — text + metadata)
from langchain_core.documents import Document

docs = [
    Document(page_content=text, metadata={"source": "tutorial.txt", "page": 1})
]
split_docs = splitter.split_documents(docs)

# Metadata is PRESERVED and COPIED to every chunk
print(split_docs[0].metadata)  # {"source": "tutorial.txt", "page": 1}
print(split_docs[0].page_content[:100])