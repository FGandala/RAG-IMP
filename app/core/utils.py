from typing import List, Tuple
from langchain_core.documents import Document

def reciprocal_rank_fusion(results_list: List[List[Document]], k:int = 60) -> List[Tuple[Document, float]]:
    """
    Realiza o Reciprocal Rank Fusion
    Recebe uma lista com a lista de documentos que foram gerados pelas variações da query
    Retorna uma lista com os documentos e o score por ordem de score
    """

    fused_scores = {}
    doc_map = {}

    for docs in results_list:
        for rank, doc in enumerate(docs):

            doc_str = doc.page_content


            if (doc_str not in doc_map):
                doc_map[doc_str] = doc
            
            if (doc_str not in fused_scores):
                fused_scores[doc_str] = 0
            
            fused_scores[doc_str] += 1/(k + rank + 1)
    

    reranked_results = [
        (doc_map[doc_str], score)
        for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return reranked_results
        




