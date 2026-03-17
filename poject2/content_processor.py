from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ------------------ SCHEMAS ------------------

class ModerationResult(BaseModel):
    is_safe: bool = Field(description="Whether content is safe")
    categories: List[str] = Field(description="Violated categories if any")
    confidence: float = Field(description="Confidence 0.0-1.0")
    reason: str = Field(description="Brief explanation")
    severity: str = Field(description="low/medium/high/none")

class ContentEnrichment(BaseModel):
    summary: str
    tone: str
    word_count: int
    reading_level: str

# ------------------ PARSERS ------------------

mod_parser = PydanticOutputParser(pydantic_object=ModerationResult)
enrich_parser = PydanticOutputParser(pydantic_object=ContentEnrichment)

JSON_RULE = "Return ONLY valid JSON. No explanation."

# ------------------ CHAINS ------------------

mod_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Moderate content for safety.\n"
     f"{JSON_RULE}\n"
     "{format_instructions}"
    ),
    ("human", "{content}")
]).partial(format_instructions=mod_parser.get_format_instructions())

mod_chain = mod_prompt | llm | mod_parser


enrich_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Analyze content.\n"
     f"{JSON_RULE}\n"
     "{format_instructions}"
    ),
    ("human", "{content}")
]).partial(format_instructions=enrich_parser.get_format_instructions())

enrich_chain = enrich_prompt | llm | enrich_parser

# ------------------ PARALLEL ------------------

pipeline = RunnableParallel({
    "moderation": mod_chain,
    "enrichment": enrich_chain,
})

# ------------------ RUN ------------------

content = "This is a test. I hope it's safe! Also, can you summarize it and tell me the tone? hail hit__"

result = pipeline.invoke({"content": content})

# ------------------ OUTPUT ------------------

print(f"Safe: {result['moderation'].is_safe}")
print(f"Severity: {result['moderation'].severity}")
print(f"Tone: {result['enrichment'].tone}")
print(f"Summary: {result['enrichment'].summary}")