from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser  # ✅ FIXED
from pydantic import BaseModel, Field
from typing import List

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",  # safer stable model
    temperature=0.3
)

# Define expected output shape
class TechExplanation(BaseModel):
    concept: str = Field(description="The concept being explained")
    simple_definition: str = Field(description="One sentence definition")
    analogy: str = Field(description="A real-world analogy")
    code_example: str = Field(description="A short code example if applicable")
    common_mistakes: List[str] = Field(description="2-3 common mistakes to avoid")

# Parser
parser = PydanticOutputParser(pydantic_object=TechExplanation)

# Prompt (stronger instruction)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a tech educator.\n"
     "Return ONLY valid JSON. No extra text.\n"
     "{format_instructions}"
    ),
    ("human", "Explain: {concept}")
]).partial(format_instructions=parser.get_format_instructions())

# Chain
chain = prompt | llm | parser

# Run
result = chain.invoke({"concept": "async/await in Python"})

# Output
print(f"Concept: {result.concept}")
print(f"Definition: {result.simple_definition}")
print(f"Analogy: {result.analogy}")
print(f"Code: {result.code_example}")
print(f"Mistakes: {result.common_mistakes}")