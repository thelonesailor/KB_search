# config.py - Updated for Qdrant
import os
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# Prefer loading PERPLEXITY_API_KEY from api-key.txt if present
def _load_api_key_from_file(path: str = "api-key.txt") -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                os.environ["PERPLEXITY_API_KEY"] = key
    except FileNotFoundError:
        # File not present; fall back to environment/.env
        pass
    except Exception:
        # Silently ignore other issues to avoid breaking startup
        pass

_load_api_key_from_file()


class Settings(BaseSettings):
    # Qdrant Configuration
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field("wand_ai_documents", env="QDRANT_COLLECTION_NAME")
    vector_dimension: int = Field(384, env="VECTOR_DIMENSION")

    # Perplexity API (read from api-key.txt if available)
    perplexity_api_key: str = Field(..., env="PERPLEXITY_API_KEY")
    chat_model: str = Field("sonar-pro", env="CHAT_MODEL")
    reasoning_model: str = Field("sonar-reasoning-pro", env="REASONING_MODEL")
    # Use local embedding model instead of OpenAI (Perplexity doesn't support embeddings)
    # Change to local path if you want to avoid downloading from Hugging Face
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    # Or use a local path like: "/path/to/local/model"
    perplexity_base_url: str = Field("https://api.perplexity.ai", env="PERPLEXITY_BASE_URL")

    # PostgreSQL for metadata (optional, can use Qdrant payloads)
    # database_url: Optional[str] = Field(None, env="DATABASE_URL")

    # Evaluation (unchanged)
    # langfuse_secret_key: Optional[str] = Field(None, env="LANGFUSE_SECRET_KEY")
    # langfuse_public_key: Optional[str] = Field(None, env="LANGFUSE_PUBLIC_KEY")
    # langfuse_host: Optional[str] = Field(None, env="LANGFUSE_HOST")

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields from environment


settings = Settings()