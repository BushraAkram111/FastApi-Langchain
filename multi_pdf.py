import os
import uvicorn
import tempfile
import zipfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone_class import PineconeInsertRetrieval

# Document Loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)

from dotenv import load_dotenv

load_dotenv()

# App
app = FastAPI(title="Document Q&A")

# Initialize clients
pinecone_client = PineconeInsertRetrieval()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


class QueryInput(BaseModel):
    query: str
    namespace: str


# Upload File Endpoint
@app.post("/upload-file/")
async def upload_file(file: UploadFile, namespace: str):
    try:
        suffix = file.filename.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        if suffix == "txt":
            loader = TextLoader(temp_path)
        elif suffix == "pdf":
            loader = PyMuPDFLoader(temp_path)
        elif suffix in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif suffix in ["xls", "xlsx"]:
            loader = UnstructuredExcelLoader(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only txt, pdf, doc, docx, xls, xlsx allowed.")

      
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, chunk_overlap=200
        )
        docs = text_splitter.split_documents(docs)
        pinecone_client.insert_data_in_namespace(docs, embeddings, namespace)
        return {"message": f"File '{file.filename}' uploaded and processed successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload ZIP File Endpoint
@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...), namespace: str = "default"):
    try:
        # Ensure the file is a ZIP
        if not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Only zip files are allowed.")

        # Save the uploaded ZIP file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
        
        # Extract files from the ZIP
        extracted_files = []
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall("temp_folder")
            extracted_files = zip_ref.namelist()

        # Process each extracted file
        for extracted_file in extracted_files:
            file_path = os.path.join("temp_folder", extracted_file)
            suffix = extracted_file.split(".")[-1].lower()

            # Load document based on file type
            if suffix == "txt":
                loader = TextLoader(file_path)
            elif suffix == "pdf":
                loader = PyMuPDFLoader(file_path)
            elif suffix in ["doc", "docx"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif suffix in ["xls", "xlsx"]:
                loader = UnstructuredExcelLoader(file_path)
            else:
                continue  # Skip unsupported files
            
            # Extract documents from the loader
            docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, chunk_overlap=200
        )
        docs = text_splitter.split_documents(docs)
        pinecone_client.insert_data_in_namespace(docs, embeddings, namespace)
        return {"message": f"File '{file.filename}' uploaded and processed successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(namespace, query):
    vector_store = pinecone_client.retrieve_from_namespace(embeddings, namespace)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    formatted_docs = format_docs(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_template("""
        You are an expert assistant for document-based question answering.context is dimited with ``` question is delimited with ###.
        Use ONLY the context below to answer. If not found, say "I cannot find the answer in the provided documents."

        Question: ###{question}###
        Context: ```{context}```
        Answer:""")
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": formatted_docs, "question": query})
    return result


# Ask Question from Documents
@app.post("/document-qa/")
async def document_question_answer(input: QueryInput):
    namespace = input.namespace
    query = input.query
    try:
        answer = create_qa_chain(namespace, query)
        return JSONResponse(content={"message": answer}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006)