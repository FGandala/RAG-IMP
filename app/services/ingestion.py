import os
import shutil
import tempfile
from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool 
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.core.config import settings


class IngestionService:
    def __init__(self,  embeddings):
        """
        Recebe o modelo de embedding instânciado uma única vez
        """
        self.embeddings = embeddings

    
    async def process_document(self, file: UploadFile)-> int:
        """
        Processa o arquivo, criar as chunks salva no banco vetorial 
        """
    
        suffix = ".pdf" if file.filename.endswith(".pdf") else ".txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:

            if (suffix) == ".pdf":
                loader = PyPDFLoader(tmp_path)

            else:
                 loader = TextLoader(tmp_path)


            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 500,
                chunk_overlap = 50,
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = text_splitter.split_documents(documents)

            for chunk in chunks:
                chunk.metadata["source"] = file.filename
            

            def _process_embeddings():
                if os.path.exists(settings.FAISS_INDEX_PATH):

                    vectorstore = FAISS.load_local(
                        settings.FAISS_INDEX_PATH,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )

                    vectorstore.add_documents(chunks)
                
                else:
                    vectorstore = FAISS.from_documents(chunks, self.embeddings)
                

                vectorstore.save_local(settings.FAISS_INDEX_PATH)

            await run_in_threadpool(_process_embeddings)

            return len(chunks)
    

        finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                


            



            








        


