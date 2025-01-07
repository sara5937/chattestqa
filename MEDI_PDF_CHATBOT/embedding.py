from langchain_qdrant import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient


#from MEDI_PDF_CHATBOT.data_ingestion import Data_loading
#from MEDI_PDF_CHATBOT.model_api import load_model
from dotenv import load_dotenv
import os

import sys
from exception import customexception
from logger import logging



def download_embedding(model,document):
    from MEDI_PDF_CHATBOT.data_ingestion import Data_loading
    from MEDI_PDF_CHATBOT.model_api import load_model
   
    try:
        collection_name = "qa_medi_chatbot1"
        logging.info("Creating Chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
        logging.info("chunk creation completed...")
        texts = text_splitter.split_documents(document)
        logging.info("Extracting as plain text ...")
        plain_texts = [doc.page_content for doc in document]
        logging.info("Extracting as plain text completed ...")


        #document_chunk = text_splitter.split_documents(document)
        

        
        logging.info("loading API key...")
        load_dotenv()
        logging.info("Saving API...")
        #open_ai_api = os.getenv("OPEN_AI_API")
        logging.info("Loading Qtrant API URL,API and creating client..")
        qdrant_client = QdrantClient(
        url=os.getenv("QTRAND_URL"),
        api_key=os.getenv("QDTRAND_API")
        )

        






        
        logging.info("Creating collection...")
        collection_name = "qa_medi_chatbot1"
        vector_size = 768
        if not qdrant_client.collection_exists(collection_name):
                # Creating the collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": vector_size, "distance": "Cosine"}  
    )
        vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=model
)
        vector_store.add_texts(plain_texts)
        logging.info("Returing vectorstore...")
        #return vectorstore
        return vector_store

    except Exception as e:
        raise customexception(e,sys)