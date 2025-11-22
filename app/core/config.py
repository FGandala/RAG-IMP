from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Demonstração RAG"
  

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()