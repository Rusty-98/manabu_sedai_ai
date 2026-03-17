# Sequential: 3 LLM calls chained together

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
str_parser = StrOutputParser()

# Step 1: Summarize
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in 2-3 sentences."),
    ("human", "{text}")
])
summarize_chain = summarize_prompt | llm | str_parser

# Step 2: Translate (takes output of step 1)
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following English text to Hindi."),
    ("human", "{summary}")
])
translate_chain = translate_prompt | llm | str_parser

# Step 3: Format nicely
format_prompt = ChatPromptTemplate.from_messages([
    ("system", "Format this as a polished newsletter blurb with a catchy headline in English."),
    ("human", "{translation}")
])
format_chain = format_prompt | llm | str_parser

# Full pipeline — output of each step feeds next step
# RunnablePassthrough pipes the string through as the next key
full_pipeline = (
    {"summary": summarize_chain}
    | RunnablePassthrough.assign(translation=translate_chain)
    | RunnablePassthrough.assign(newsletter=format_chain)
)

article = """
Python 3.12 was released with significant performance improvements.
The new version includes a 5% speed boost, better error messages,
and improved f-string parsing. Developers can now use type parameter
syntax for generic classes and functions.
"""

result = full_pipeline.invoke({"text": article})
print("=== Summary ===")
print(result["summary"])
print("\n=== Hindi Translation ===")
print(result["translation"])
print("\n=== Newsletter ===")
print(result["newsletter"])