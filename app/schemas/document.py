from pydantic import BaseModel, Field
from typing import List, Optional


class RetrievalRequest(BaseModel):
    """
    Esquema que representa uma query do usuário
    """
    query: str = Field(..., min_lenght=3, max_length=1000, description="A query do usuário para a busca vetorial")
    k: int = Field(default=4, ge=1, le=20, description = "Número de documentos a serem retornados")


class DocumentResponse(BaseModel):
    """
    Esquema que representa um único documento
    """
    content:str
    source:str
    page: Optional[int] = None
    score: Optional[float] = None


class RetrievalResponse(BaseModel):
    """
    Esquema que representa a resposta da busca pelos documentos
    """
    results:List[DocumentResponse]

class IngestionResponse(BaseModel):
    """
    Esquema que representa a resposta da ingestão do documento
    """
    filename: str
    status: str
    chunks_created: int
    message: str
