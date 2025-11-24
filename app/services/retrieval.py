import os
from typing import List
from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import FAISS
from app.core.config import settings
from app.schemas.document import DocumentResponse
from app.core.utils import reciprocal_rank_fusion
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RetrievalService():
    def __init__(self, embeddings):
        """
        Recebe o modelo de embedding instânciado uma única vez, inicializa o groq e a query inicial para o rag-fusion
        """
        self.embedding = embeddings
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.5,
        )

        self.fusion_query = ChatPromptTemplate.from_template(
            "Você é um assistente especialista em busca semântica. \n"
            "Sua tarefa é gerar 5 versões diferentes da pergunta do usuário para "
            "maximizar as chances de encontrar a resposta correta em um banco vetorial.\n"
            "Regras:\n"
            "1. Gere variações que busquem diferentes ângulos ou sinônimos.\n"
            "2. Forneça apenas as perguntas, uma por linha.\n"
            "3. Não numere as linhas e não use bullets.\n"
            "Pergunta original: {question}"
        )
        
    async def search(self, query: str, k:int = 4 )->List[DocumentResponse]:
        """
        Faz a busca por similaridade e retorna uma lista com as respostas
        """

        if not os.path.exists(settings.FAISS_INDEX_PATH):
            return []
        

        try:
            chain = self.fusion_query | self.llm | StrOutputParser()

            generated_text = await chain.ainvoke({"question": query})

            variations = [line.strip() for line in generated_text.split("\n") if line.strip()]

            all_queries = variations + [query]

            print(f"Queries Geradas: {all_queries}")
        
        except Exception as e:
            print(f"Erro ao gerar variações no Groq: {e}. Usando apenas query original.")
            all_queries = [query]


        def _search_logic():

            vectorstore = FAISS.load_local(
                settings.FAISS_INDEX_PATH,
                self.embedding,
                allow_dangerous_deserialization=True
            )
            results_list = []

            for q in all_queries:
                docs = vectorstore.similarity_search(q, k=k)
                results_list.append(docs)

            return results_list
        
        raw_results = await run_in_threadpool(_search_logic)

        reranked_results = reciprocal_rank_fusion(raw_results)

        response_list = []

        for doc, new_score in reranked_results[:k]:
            response_list.append(DocumentResponse(
                content=doc.page_content,
                source=doc.metadata.get("source", "desconhecido"),
                page=doc.metadata.get("page"),
                score=float(new_score) 
            ))

        return response_list

