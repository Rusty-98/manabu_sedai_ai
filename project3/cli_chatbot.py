from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
You remember the conversation and give contextual responses.
Keep responses concise unless asked to elaborate."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# In-memory store
store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    hist = store[session_id]
    # Window: keep last 20 messages
    if len(hist.messages) > 20:
        hist.messages = hist.messages[-20:]
    return hist

chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

session_id = "main"

print("Chatbot ready! Type 'quit' to exit, 'history' to see messages.\n")
while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "history":
        for msg in store.get(session_id, ChatMessageHistory()).messages:
            print(f"  [{msg.type}]: {msg.content[:80]}...")
        continue

    print("AI: ", end="", flush=True)
    config = {"configurable": {"session_id": session_id}}
    for chunk in chatbot.stream({"input": user_input}, config=config):
        print(chunk, end="", flush=True)
    print()