from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
response = llm.invoke("Say hello in one sentence.")
print(response.content[0]['text'])
