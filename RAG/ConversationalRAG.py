# RAG + memory (the full production pattern)

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 1: Make retriever history-aware
# "What is it used for?" needs history to know "it" = Python
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and a new question, rewrite the question to be standalone (no pronouns referring to history). Return ONLY the rewritten question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# Step 2: QA chain with context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context below.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# Step 3: Full conversational RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Step 4: Add memory management
store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

config = {"configurable": {"session_id": "rag_session"}}

# Multi-turn conversation with context
turns = [
    "What is Python?",
    "When was it created?",       # "it" = Python — needs history
    "What about version 3.12?",   # still about Python
]
for q in turns:
    result = conversational_rag.invoke({"input": q}, config=config)
    print(f"Q: {q}\nA: {result['answer']}\n")