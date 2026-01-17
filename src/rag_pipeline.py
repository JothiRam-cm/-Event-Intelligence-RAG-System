from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompt import get_rag_prompt
from src.retriever import get_relevant_events
import os

from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def run_rag(query: str):
    # 1. Retrieve events
    docs = get_relevant_events(query)

    if not docs:
        return "No relevant event data found for the given question."

    # 2. Build context
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. Prompt + LLM
    prompt = get_rag_prompt()
    llm = get_llm()

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    return response.content
