from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.3)

# Multiple specialized chains
chains = {
    "explain": ChatPromptTemplate.from_messages([
        ("system", "Explain concepts clearly with examples."),
        ("human", "{input}")
    ]) | llm | StrOutputParser(),

    "debug": ChatPromptTemplate.from_messages([
        ("system", "You are a debugging expert. Find the bug and explain the fix."),
        ("human", "Debug this code:\n{input}")
    ]) | llm | StrOutputParser(),

    "review": ChatPromptTemplate.from_messages([
        ("system", "Review code for best practices, performance, security."),
        ("human", "Review this code:\n{input}")
    ]) | llm | StrOutputParser(),
}

print("Commands: explain / debug / review / quit")

while True:
    cmd = input("\nCommand: ").strip().lower()
    if cmd == "quit":
        break
    if cmd not in chains:
        print(f"Unknown command. Use: {list(chains.keys())}")
        continue
    user_input = input("Input: ").strip()
    print("\n--- Response ---")
    # Stream the response token by token
    for chunk in chains[cmd].stream({"input": user_input}):
        print(chunk, end="", flush=True)
    print()