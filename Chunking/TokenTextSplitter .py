from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter

text = """
## Python
Python is a high-level programming language. It is easy to read and write.
It supports object-oriented programming, functional programming, and procedural programming.
Python is widely used in web development, data science, automation, and machine learning.

## Machine Learning
Machine learning is a branch of artificial intelligence.
It allows systems to learn from data instead of being explicitly programmed.
Supervised learning, unsupervised learning, and reinforcement learning are common types.
Applications include recommendation systems, fraud detection, and image recognition.

## Web Development
Web development involves building websites and web applications.
Frontend development focuses on what users see and interact with.
Backend development handles servers, databases, and application logic.
Popular tools include React, Node.js, Express, and MongoDB.

## Databases
Databases are used to store and manage data efficiently.
SQL databases include MySQL and PostgreSQL.
NoSQL databases include MongoDB and Firebase.
Choosing the right database depends on project requirements.
"""

# Token-based splitter
splitter = TokenTextSplitter(
    chunk_size=80,
    chunk_overlap=10
)

chunks = splitter.split_text(text)

print(f"Token Chunks: {len(chunks)}")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Token Chunk {i} ---")
    print(chunk)

print("\nEach chunk is approximately within the token limit you set.")

# Character-based splitter for markdown-like text
md_splitter = CharacterTextSplitter(
    separator="\n## ",
    chunk_size=200,
    chunk_overlap=0,
)

md_chunks = md_splitter.split_text(text)

print(f"\nMarkdown-style chunks: {len(md_chunks)}")
for i, chunk in enumerate(md_chunks, 1):
    print(f"\n--- Markdown Chunk {i} ---")
    print(chunk)