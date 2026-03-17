from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

# Sync streaming
print("Response: ", end="", flush=True)

for chunk in llm.stream("Explain decorators in Python in 3 sentences."):
    if not chunk.content:
        continue

    for part in chunk.content:
        if part.get("type") == "text":
            print(part.get("text", ""), end="", flush=True)

print()

# Async streaming (use in FastAPI)
async def stream_response(question: str):
    async for chunk in llm.astream(question):
        if not chunk.content:
            continue

        for part in chunk.content:
            if part.get("type") == "text":
                yield part.get("text", "")

# Key methods summary:
# llm.invoke()   → sync, returns full response
# llm.ainvoke()  → async, returns full response
# llm.stream()   → sync, yields chunks
# llm.astream()  → async, yields chunks (use in FastAPI)
# llm.batch()    → sync, list in → list out (parallel)