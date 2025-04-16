from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
load_dotenv()

 
class PineconeInsertRetrieval:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = "test-index"
    
    # Create New nameSpace and insert Data in it
    def insert_data_in_namespace(self,documents,embeddings,name_space):
        try:
            doc_search=PineconeVectorStore.from_documents(
                documents,
                embeddings,
                index_name=self.index_name,
                namespace = name_space
                )
            print(f"Your Name space {name_space} is Created successfully")
            return doc_search
        except Exception as ex:
            return f"Failed to created namespace {ex}"
   

       
    # Retrieve Data from Namespace
    def retrieve_from_namespace(self,embeddings,name_space):
        try:
            vectorstore = PineconeVectorStore.from_existing_index(
                embedding=embeddings,index_name=self.index_name,namespace=name_space)
            return vectorstore
        except Exception as ex:
            return f"Failed to load VectorStore {ex}"
 
 
pine_ = PineconeInsertRetrieval()
 