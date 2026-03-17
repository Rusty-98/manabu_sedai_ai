from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

# Each piece individually
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
parser = StrOutputParser()   # converts AIMessage → plain string

# Chain them with pipe
chain = prompt | llm | parser

# .invoke() runs the whole chain
result = chain.invoke({"question": "What is Python?"})
print(type(result))   # str  ← not AIMessage, because of parser
print(result)

# What actually happens internally:
# 1. prompt.invoke({"question": "..."}) → ChatPromptValue
# 2. llm.invoke(ChatPromptValue)        → AIMessage
# 3. parser.invoke(AIMessage)           → str

# Async version — same chain, just await it
# result = await chain.ainvoke({"question": "What is Python?"})
# print(result)