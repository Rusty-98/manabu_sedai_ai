# RULE 1: Start with RecursiveCharacterTextSplitter, tune from there
# It works well for 80% of use cases out of the box

# RULE 2: chunk_size depends on your content
# Short factual docs (FAQs, product specs)  → chunk_size=300-500
# Long narrative docs (articles, books)     → chunk_size=800-1200
# Code files                                → chunk_size=1000-2000
# Legal/technical docs                      → chunk_size=500-800

# RULE 3: overlap = ~15-20% of chunk_size
# chunk_size=500  → overlap=75-100
# chunk_size=1000 → overlap=150-200

# RULE 4: Always add metadata to chunks
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_with_metadata(text: str, source: str, page: int = 1):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=75
    )
    doc = Document(
        page_content=text,
        metadata={
            "source": source,    # filename or URL
            "page": page,        # page number
            "chunk_index": 0,    # will be updated below
        }
    )
    chunks = splitter.split_documents([doc])
    # Add chunk index to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    return chunks