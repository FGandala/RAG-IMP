import os
from typing import List
from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import FAISS
from app.core.config import settings
from app.schemas.document import DocumentResponse

class RetrievalService():
    def __init__(self, embeddings):
        """
        Recebe o modelo de embedding instânciado uma única vez
        """
        self.embedding = embeddings
        pass
    async def search(self, query: str, k:int = 4 )->List[DocumentResponse]:
        """
        Faz a busca por similaridade e retorna uma lista com as respostas
        """

        if not os.path.exists(settings.FAISS_INDEX_PATH):
            return []
        
        def _search_logic():

            vectorstore = FAISS.load_local(
                settings.FAISS_INDEX_PATH,
                self.embedding,
                allow_dangerous_deserialization=True
            )

            return vectorstore._similarity_search_with_relevance_scores(query, k=k)
        
        raw_results = await run_in_threadpool(_search_logic)

        reponse_list = []

        for doc, score in raw_results:
            reponse_list.append(
                DocumentResponse(
                    content=doc.page_content,
                    source=doc.metadata.get("source","desconhecido"),
                    page=doc.metadata.get("page"),
                    score=float(score)
                )
            )
        return reponse_list

