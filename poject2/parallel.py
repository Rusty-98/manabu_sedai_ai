from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.3
)

# Common rule
JSON_RULE = "Return ONLY valid JSON. No explanation."

# --- Sentiment ---
sentiment_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         JSON_RULE +
         '\n{{"sentiment": "positive/negative/neutral", "score": 0.0}}'
        ),
        ("human", "{text}")
    ])
    | llm
    | JsonOutputParser()
)

# --- Keywords ---
keywords_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         JSON_RULE +
         '\n{{"keywords": ["k1", "k2", "k3", "k4", "k5"]}}'
        ),
        ("human", "{text}")
    ])
    | llm
    | JsonOutputParser()
)

# --- Category ---
category_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         JSON_RULE +
         '\n{{"category": "tech/business/sports/health/other", "confidence": 0.0}}'
        ),
        ("human", "{text}")
    ])
    | llm
    | JsonOutputParser()
)

# --- Parallel execution ---
analysis_pipeline = RunnableParallel({
    "sentiment": sentiment_chain,
    "keywords": keywords_chain,
    "category": category_chain,
    "original_text": RunnableLambda(lambda x: x["text"]),
})

text = "Apple released new MacBook Pro models with M3 chips offering significant performance gains for developers."

result = analysis_pipeline.invoke({"text": text})

# --- Output ---
print(f"Sentiment: {result['sentiment']['sentiment']} ({result['sentiment']['score']})")
print(f"Category: {result['category']['category']} ({result['category']['confidence']})")
print(f"Keywords: {result['keywords']['keywords']}")