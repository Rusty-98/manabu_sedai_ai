from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Convert vectorstore to retriever
# Retriever has one method: .invoke(query) → List[Document]

# 1. Basic retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # cosine similarity
    search_kwargs={"k": 4}     # return top 4
)
docs = retriever.invoke("What is RAG?")

# 2. MMR — Maximal Marginal Relevance
# Returns diverse results, not just the most similar ones
# Avoids getting 4 chunks that all say the same thing
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,            # return 4 docs
        "fetch_k": 20,     # consider top 20 first
        "lambda_mult": 0.7 # 1=pure similarity, 0=pure diversity
    }
)

# 3. Score threshold — only return high-confidence matches
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5}
)

# All retrievers have the same interface:
# retriever.invoke(query)   → sync
# retriever.ainvoke(query)  → async
# retriever.batch([q1, q2]) → multiple queries

# They also plug directly into LCEL chains:
# chain = retriever | format_docs | prompt | llm | parser