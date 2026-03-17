from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

# 1. Simple string invoke — quickest
result = llm.invoke("What is LangChain?")
print(result.content[0]['text'])          # the text response
print(result.usage_metadata)   # token counts

# 2. With message objects — more control
messages = [
    SystemMessage(content="You are a Python expert. Be concise."),
    HumanMessage(content="What is a decorator?")
]
result = llm.invoke(messages)
print(result.content[0]['text'])

# 3. Batch — multiple inputs at once (runs in parallel)
questions = ["What is Python?", "What is LangChain?", "What is RAG?"]
results = llm.batch(questions)
for r in results:
    print(r.content[0]['text'][:100])