# llm_providers.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

def get_llm_from_provider():
    """
    Initializes and returns an LLM instance based on the provider
    specified in the .env file.
    """
    provider = os.getenv("LLM_PROVIDER")
    model_name = os.getenv("MODEL_NAME")

    if not provider or not model_name:
        raise ValueError(
            "LLM_PROVIDER and MODEL_NAME must be set in the .env file"
        )

    print(f"Provider: {provider}, Model: {model_name}")

    if provider.lower() == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        return ChatGroq(api_key=api_key, model_name=model_name)

    elif provider.lower() == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    elif provider.lower() == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")