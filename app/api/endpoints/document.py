from fastapi import APIRouter, UploadFile, File, HTTPException, status
from langchain_huggingface import HuggingFaceEmbeddings
from app.schemas.document import RetrievalRequest, RetrievalResponse, IngestionResponse, DocumentResponse
from app.services.ingestion import IngestionService
from app.core.config import settings


try:
    global_embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={
            'device': 'cpu', 
            'trust_remote_code': True,
            'token': settings.HF_TOKEN
        },
        encode_kwargs={'normalize_embeddings': True}
    )

    ingestion_service = IngestionService(embeddings=global_embeddings)
    
    print("Modelo carregado e serviços prontos!")

except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    raise e


router = APIRouter()


@router.post("/ingest", response_model=IngestionResponse,status_code=status.HTTP_201_CREATED)
async def ingest_document(file: UploadFile = File(...)):
    """
    Recebe um arquivo (PDF/TXT), processa e indexa no banco vetorial.
    """
    if(file.content_type not in ["application/pdf","text/plain"]):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF ou TXT são permitidos.")
    
    try:
        
        number_of_chunks = await ingestion_service.process_document(file)

        return IngestionResponse(
        filename=file.filename,
        status="success",
        chunks_created=number_of_chunks,
        message="Documento processado e indexado com sucesso."
    )

    except Exception as e:

        print(f"Erro na Ingestão: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erro interno ao processar documento: {str(e)}"
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



