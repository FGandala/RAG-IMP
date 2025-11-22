from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Demonstração RAG"

    EMBEDDING_MODEL_NAME: str = "google/embeddinggemma-300m"

    HF_TOKEN: str

    FAISS_INDEX_PATH: str = "faiss_index_store"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()