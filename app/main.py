from fastapi import FastAPI
from app.core.config import settings
from app.api.endpoints import document

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="RAG-API",
    version="1.0.0"
)

app.include_router(document.router, prefix="/api", tags=["Documents"])

@app.get("/")
async def root():
    return {"message": "Bem vindo a API de demonstração do RAG",}