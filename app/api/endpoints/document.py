import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from langchain_huggingface import HuggingFaceEmbeddings
from app.schemas.document import RetrievalRequest, RetrievalResponse, IngestionResponse
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.core.config import settings
from huggingface_hub import login


try:
    login(token=settings.HF_TOKEN)
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
    retrieval_service = RetrievalService(embeddings=global_embeddings)
    
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
    query = request.query
    k = request.k
    results = await retrieval_service.search(query, k)

    
    return RetrievalResponse(results=results)


@router.delete("/reset", status_code=status.HTTP_200_OK)
async def reset_knowledge_base():
    """
    Limpa todos os documentos ingeridos
    """

    folder_path = settings.FAISS_INDEX_PATH

    try:
        if(os.path.exists(folder_path)):

            shutil.rmtree(folder_path)
            return {"status": "success", "message": "Banco vetorial resetado com sucesso."}
        
        else:
            return {"status": "warning", "message": "O banco já estava vazio."}
    
    except Exception as e:
        print("Erro {e} ao tentar deletar o banco vetorial")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro {e} ao tentar deletar o banco vetorial"
        )






