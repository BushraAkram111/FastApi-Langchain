from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_openai import OpenAIEmbeddings 
import os

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore , Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever

from langchain.document_loaders import TextLoader


chunk_size = 500
chunk_overlap = 50


def load_split_text_file(file):
    loader = TextLoader(file)
    documents = loader.load()
    documents_content = [doc.page_content for doc in documents]

    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    # Split the text into smaller chunks
    texts = text_splitter.create_documents(documents_content)

    return texts

def load_split_pdf_file(file):
    loader = PyMuPDFLoader(file)
    pages = loader.load()
    pdf_page_content = [page.page_content for page in pages]

    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function=len,
        #separators= ["\n\n", "\n", " ", ""]
    )

    # Split the text into smaller chunks
    chunks = text_splitter.create_documents(pdf_page_content)


    return chunks

def load_split_docx_file(file):
    loader = UnstructuredWordDocumentLoader(str(file))
    documents = loader.load()
    word_docx_content = [doc.page_content for doc in documents]
    textsplit = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function=len)

    doc_chunks = textsplit.create_documents(word_docx_content)

    return doc_chunks

class QdrantInsertRetrievalAll:
    def __init__(self,api_key,url):
        self.url = url 
        self.api_key = api_key

    # Method to insert documents into Qdrant vector store
    def insertion(self,text,embeddings,collection_name):
        qdrant = QdrantVectorStore.from_documents(
        text,
        embeddings,
        url=self.url,
        prefer_grpc=True,
        api_key=self.api_key,
        collection_name=collection_name,
        force_recreate=True
        )
        print("insertion successfull")
        return qdrant

    # Method to retrieve documents from Qdrant vector store
    def retrieval(self,collection_name,embeddings):
        qdrant_client = QdrantClient(
        url=self.url,
        api_key=self.api_key,
        )
        qdrant_store = Qdrant(qdrant_client,collection_name=collection_name ,embeddings=embeddings)
        return qdrant_store
    

def QA_Chain_Retrieval(query, qdrant_vectordb):
    try:
        # Formatting function for documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Prompt template string
        prompt_str = """
        Answer the user question based only on the following context:
        {context}

        Question: {question}
        """
        
        # Create a chat prompt template
        _prompt = ChatPromptTemplate.from_template(prompt_str)
        
        # Set the number of chunks to retrieve
        num_chunks = 3
        
        # Set up the retriever
        retriever = qdrant_vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_chunks}
        )
        
        # Set up the chain components
        chat_llm = ChatOpenAI(model_name="gpt-4o-mini")
        query_fetcher = itemgetter("question")
        setup = {
            "question": query_fetcher,
            "context": query_fetcher | retriever | format_docs
        }
        
        # Define the final chain
        _chain = setup | _prompt | chat_llm
        
        # Execute the chain and fetch the response
        response = _chain.invoke({"question": query})
        return response
    
    except Exception as e:
        return f"Error executing retrieval chain: {str(e)}"
