from langchain_google_genai import ChatGoogleGenerativeAI
import os


def get_llm():
    """
    Returns the LLM used for RAG answers.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
