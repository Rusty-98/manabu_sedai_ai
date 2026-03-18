from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel
from typing import List

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

docs = [
    Document(page_content="Python was created by Guido van Rossum in 1991.", metadata={"source": "python_history.txt", "page": 1}),
    Document(page_content="Python is used for web dev, data science, AI, automation.", metadata={"source": "python_uses.txt", "page": 1}),
    Document(page_content="Python 3.12 introduced significant performance improvements.", metadata={"source": "python_versions.txt", "page": 3}),
]
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Chain that returns BOTH answer AND source docs
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only this context:\n\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(f"[{d.metadata.get('source','unknown')}]\n{d.page_content}" for d in docs)

# Parallel chain: run retrieval once, use result for both answer + sources
rag_with_sources = RunnableParallel({
    "docs": retriever,                           # raw Document objects
    "answer": (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt | llm | StrOutputParser()
    )
})

result = rag_with_sources.invoke("When was Python created?")

print(f"Answer: {result['answer']}\n")
print("Sources:")
for doc in result["docs"]:
    print(f"  - {doc.metadata.get('source')} (page {doc.metadata.get('page', '?')})")
    print(f"    {doc.page_content[:80]}...")