from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Sample knowledge base
docs = [
    Document(page_content="RAG stands for Retrieval Augmented Generation. It combines vector search with LLM generation to answer questions from your own documents."),
    Document(page_content="Chroma is an open-source vector database. It stores embeddings locally and supports similarity search using cosine distance."),
    Document(page_content="Gemini embeddings use the text-embedding-004 model which produces 768-dimensional vectors. Use retrieval_document task_type when indexing."),
    Document(page_content="Chunking splits documents into smaller pieces before embedding. RecursiveCharacterTextSplitter is the most common splitter in LangChain."),
    Document(page_content="Cosine similarity measures the angle between two vectors. A score of 1.0 means identical meaning, 0.0 means completely unrelated."),
]

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG prompt — context + question
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY
the provided context. If the answer is not in the context, say
"I don't have that information in my knowledge base."

Context:
{context}"""),
    ("human", "{question}")
])

# Helper: format retrieved docs into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# THE RAG CHAIN
# RunnablePassthrough() passes the question through unchanged
# retriever fetches relevant docs, format_docs joins them
rag_chain = (
    {
        "context": retriever | format_docs,  # retrieve + format
        "question": RunnablePassthrough()    # pass question as-is
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Ask questions
questions = [
    "What is RAG?",
    "What dimensions does Gemini embeddings produce?",
    "What is the best chunking strategy?",
    "What is the capital of France?"  # not in context
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {rag_chain.invoke(q)}\n")