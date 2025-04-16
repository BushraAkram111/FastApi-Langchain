import os
import uvicorn
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
# Langchain & Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_class import QdrantInsertRetrievalAll

# Document Loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
)



from dotenv import load_dotenv

load_dotenv()

# App
app = FastAPI(title="Document Q&A")

# Initialize clients
qdrant_client = QdrantInsertRetrievalAll()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")




class QueryInput(BaseModel):
    query: str
    collection_name: str


# Upload File Endpoint
@app.post("/upload-file/")
async def upload_file(file: UploadFile, collection_name: str):
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
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only txt, pdf, doc, docx allowed.")

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, chunk_overlap=200
        )
        docs = text_splitter.split_documents(docs)
        qdrant_client.insertion(docs, embeddings, collection_name)
        return {"message": f"File '{file.filename}' uploaded and processed successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(collection_name, query):
    vector_store = qdrant_client.retrieval(collection_name, embeddings)
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
    collection_name = input.collection_name
    query = input.query
    try:
        answer = create_qa_chain(collection_name, query)
        return JSONResponse(content={"message": answer}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006)