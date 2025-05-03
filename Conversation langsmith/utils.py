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

# ✅ LangSmith (Tracing) Setup
from langsmith import Client
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# 🔷 CLASS: Handles insertion and retrieval from Qdrant vector DB
class QdrantInsertRetrievalAll:
    def __init__(self, api_key, url):
        self.url = url 
        self.api_key = api_key

    def insertion(self, text, embeddings, collection_name):
        qdrant = QdrantVectorStore.from_documents(
            text,
            embeddings,
            url=self.url,
            prefer_grpc=True,
            api_key=self.api_key,
            collection_name=collection_name,
            force_recreate=True
        )
        print("insertion successful")
        return qdrant

    def retrieval(self, collection_name, embeddings):
        qdrant_client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        qdrant_store = Qdrant(qdrant_client, collection_name=collection_name, embeddings=embeddings)
        return qdrant_store


# 🔷 FUNCTION: Builds the conversational response using user query + chat history
def Conversational_Chain(query, history):
    try:
        template = """You are an expert chatbot assistant. You also have access to the user's conversation history. 
        Answer the user's question based on this history.
        
        history: {HISTORY}
        query: {QUESTION}
        """
        prompt = ChatPromptTemplate.from_template(template)

        model = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            verbose=True  # ✅ Enable LangSmith tracing & local debugging
        )

        setup = RunnableParallel({
            "HISTORY": RunnablePassthrough(),
            "QUESTION": RunnablePassthrough()
        })

        output_parser = StrOutputParser()

        rag_chain = (
            setup
            | prompt
            | model
            | output_parser
        )

        input_dict = {
            "QUESTION": query,
            "HISTORY": history
        }

        # ✅ LangSmith tracing with custom run name
        response = rag_chain.invoke(
            str(input_dict),
            config={"run_name": "Conversational_Chain"}
        )
        return response

    except Exception as e:
        return f"Error executing conversational retrieval chain: {str(e)}"
