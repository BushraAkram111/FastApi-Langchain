from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize environment and qdrant

api_key = os.getenv("qdrant_api_key")
url = os.getenv("qdrant_url")

class QdrantInsertRetrievalAll:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=url,
            api_key=api_key
        )

    # Method to insert documents into Qdrant vector store
    def insertion(self, docs, embeddings, collection_name):
        qdrant = QdrantVectorStore.from_documents(
            docs,
            embeddings,
            url=url,
            api_key=api_key,
            prefer_grpc=True,
            collection_name=collection_name,
        )
        return qdrant

    # Method to retrieve documents from Qdrant vector store
    def retrieval(self, collection_name, embeddings):
        # Create vector store with client
        qdrant_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        return qdrant_store