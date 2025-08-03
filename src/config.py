# src/config.py
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the AI pipeline."""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENWEATHERMAP_API_KEY: str = os.getenv("OPENWEATHERMAP_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    
    # Qdrant Cloud Settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = "pdf_embeddings"
    
    # LangSmith Settings
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "ai-pipeline-demo")
    
    # OpenWeather API
    WEATHER_BASE_URL: str = "http://api.openweathermap.org/data/2.5/weather"
    
    # Groq API Settings
    MODEL_NAME: str = "llama3-8b-8192"
    
    # Embedding model (using sentence-transformers)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        required_keys = ["GROQ_API_KEY", "OPENWEATHERMAP_API_KEY"]
        missing = [key for key in required_keys if not getattr(cls, key)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        # Validate Qdrant configuration
        if not cls.QDRANT_URL or not cls.QDRANT_API_KEY:
            print("Warning: Qdrant Cloud credentials not configured. Using in-memory fallback.")

# Initialize configuration
try:
    Config.validate()
except ValueError as e:
    print(f"Warning: {e}")