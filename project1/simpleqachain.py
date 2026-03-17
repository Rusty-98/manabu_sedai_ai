from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Model
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.3)

# 2. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer clearly and concisely."),
    ("human", "{question}")
])

# 3. Parser
parser = StrOutputParser()

# 4. Chain — the pipe connects all 3
chain = prompt | llm | parser

# 5. Run it
result = chain.invoke({
    "role": "Python expert",
    "question": "What is the difference between a list and a tuple?"
})
print(result)

# Try different roles
result2 = chain.invoke({
    "role": "5-year-old teacher",
    "question": "What is a database?"
})
print(result2)