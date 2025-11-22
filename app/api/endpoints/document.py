from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.document import RetrievalRequest, RetrievalResponse, IngestionResponse, DocumentResponse

router = APIRouter()

@router.post("/ingest", response_model=IngestionResponse,status_code=status.HTTP_201_CREATED)
async def ingest_document(file: UploadFile = File(...)):
    """
    Recebe um arquivo (PDF/TXT), processa e indexa no banco vetorial.
    """
    if(file.content_type not in ["application/pdf","text/plain"]):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF ou TXT são permitidos.")
    

    return IngestionResponse(
        filename=file.filename,
        status="success",
        chunks_created=10,
        message="Documento processado e indexado com sucesso."
    )


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_document(request: RetrievalRequest):
    """
    Realiza a busca vetorial baseada na pergunta do usuário.
    """
    mock_docs = [
        DocumentResponse(content="O FAISS é uma biblioteca...", source="manual.pdf", page=1, score=0.95),
        DocumentResponse(content="LangChain facilita a integração...", source="docs.txt", page=None, score=0.88)
    ]
    
    return RetrievalResponse(results=mock_docs)



