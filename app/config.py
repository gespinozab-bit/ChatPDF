import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5433/week_8",
    )
    collection_name: str = os.getenv("PGVECTOR_COLLECTION", "pdf_chat_documents")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    chunk_size: int = _env_int("CHUNK_SIZE", 1000)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 200)
    default_top_k: int = _env_int("TOP_K", 4)
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 20)


settings = Settings()
