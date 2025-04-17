# Loaders for PDF, Word, and text splitting (LangChain)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import TextLoader

# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings 
import os

# Qdrant vector database imports
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, Qdrant
from qdrant_client import QdrantClient, models

# LangChain Core Modules
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter

# Runnable classes for building pipelines
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# For compressing retrieved documents (optional advanced feature)
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever


# 🔷 CLASS: Handles insertion and retrieval from Qdrant vector DB
class QdrantInsertRetrievalAll:
    def __init__(self, api_key, url):
        # Set the Qdrant URL and API key
        self.url = url 
        self.api_key = api_key

    # ✅ Insert documents into Qdrant vector store
    def insertion(self, text, embeddings, collection_name):
        qdrant = QdrantVectorStore.from_documents(
            text,  # List of documents (already split/cleaned)
            embeddings,  # Embedding model to use
            url=self.url,  # Qdrant cloud/local URL
            prefer_grpc=True,  # Use GRPC protocol for speed
            api_key=self.api_key,  # Authentication
            collection_name=collection_name,  # Unique name for this collection
            force_recreate=True  # Recreate the collection if it already exists
        )
        print("insertion successful")
        return qdrant  # Return the Qdrant vector store object

    # ✅ Retrieve documents from Qdrant vector store
    def retrieval(self, collection_name, embeddings):
        qdrant_client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        # Create a retriever object using the Qdrant store
        qdrant_store = Qdrant(qdrant_client, collection_name=collection_name, embeddings=embeddings)
        return qdrant_store


# 🔷 FUNCTION: Builds the conversational response using user query + chat history
def Conversational_Chain(query, history):
    try:
        # Define a prompt template with both history and user query
        template = """You are an expert chatbot assistant. You also have access to the user's conversation history. 
        Answer the user's question based on this history.
        
        history: {HISTORY}
        query: {QUESTION}
        """
        # Create a LangChain prompt object
        prompt = ChatPromptTemplate.from_template(template)

        # Initialize the OpenAI Chat model (GPT-4o-mini)
        model = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0  # Deterministic responses
        )

        # Create a pipeline that passes both HISTORY and QUESTION to the prompt
        setup = RunnableParallel({
            "HISTORY": RunnablePassthrough(),
            "QUESTION": RunnablePassthrough()
        })

        # This will parse the LLM's output to a plain string
        output_parser = StrOutputParser()

        # Combine all parts: input → prompt → model → output parser
        rag_chain = (
            setup
            | prompt
            | model
            | output_parser
        )

        # Prepare the actual inputs
        input_dict = {
            "QUESTION": query,
            "HISTORY": history
        }

        # Invoke the chain with the inputs and get the response
        response = rag_chain.invoke(str(input_dict))
        return response

    except Exception as e:
        # Return error message if something fails
        return f"Error executing conversational retrieval chain: {str(e)}"
