from fastapi import FastAPI, File, UploadFile, Form
from utils import QdrantInsertRetrievalAll, Conversational_Chain
import tempfile
import os
from langchain_openai import OpenAIEmbeddings
import uvicorn
from pydantic import BaseModel
from conv_ret_db import SessionLocal, ConversationChatHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from operator import itemgetter
import ast
from fastapi.responses import JSONResponse
from fastapi import status
from dotenv import load_dotenv

# ✅ LangSmith Imports
from langsmith import Client
from dotenv import find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

client = Client()
dataset_name = "conversation_dataset"
try:
    dataset = client.create_dataset(dataset_name)
except:
    dataset = client.get_dataset(dataset_name=dataset_name)

# OpenAI and Embedding setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
Embeddings_model = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=Embeddings_model, api_key=OPENAI_API_KEY)

# FastAPI App
app = FastAPI()

# Input model
class QueryRequest(BaseModel):
    chatbot_id: str
    query: str

# ✅ 1. Chatbot conversation endpoint
@app.post("/Conversation")
async def Convo_chain(request: QueryRequest):
    chatbot_id_user = request.chatbot_id
    query_user = request.query

    session = SessionLocal()
    try:
        chatbot_id_table = session.query(ConversationChatHistory).filter_by(chatbot_id=chatbot_id_user).first()
        chatbot_id = chatbot_id_table.chatbot_id if chatbot_id_table else chatbot_id_user

        conversation_history = session.query(ConversationChatHistory).filter_by(chatbot_id=chatbot_id).order_by(ConversationChatHistory.id.desc()).limit(30).all()

        chat_history = []
        for chat in conversation_history:
            if chat.query:
                chat_history.append(f"User: {chat.query}")
            if chat.response:
                chat_history.append(f"AI response: {chat.response}")

        llm_model = ChatOpenAI(model='gpt-4o-mini', openai_api_key=OPENAI_API_KEY, temperature=0, verbose=True)
        greeting_prompt = """
            You are a smart AI assistant. Your task is to classify whether the given user query is a greeting (like 'Hi', 'Hello', etc.).
            If the query is a greeting, respond with: ["yes", "your greeting response"]
            If the query is not a greeting, respond with: ["no", "statusCode:404"]
            User query: {query}
            Make sure to follow the exact output format: ["yes/no", "greeting_response"]
        """
        greeting_prompt = ChatPromptTemplate.from_template(greeting_prompt)
        greeting_chain = ({"query": itemgetter("query")} | greeting_prompt | llm_model)
        greeting_response = greeting_chain.invoke({"query": query_user}, config={"run_name": "Greeting_Chain"})
        response = ast.literal_eval(greeting_response.content)
        label = response[0]

        if label.lower().endswith("yes"):
            message = response[1]
            session.add(ConversationChatHistory(chatbot_id=chatbot_id, query=query_user, response=message))
            session.commit()

            # ✅ Save to LangSmith dataset
            client.create_example(
                inputs={"query": query_user},
                outputs={"response": message},
                dataset_id=dataset.id
            )

            return JSONResponse(content={"message": "Response Generated Successfully!", "data": message}, status_code=status.HTTP_200_OK)

        results = Conversational_Chain(query=query_user, history=chat_history)
        session.add(ConversationChatHistory(chatbot_id=chatbot_id, query=query_user, response=results))
        session.commit()

        # ✅ Save to LangSmith dataset
        client.create_example(
            inputs={"query": query_user},
            outputs={"response": results},
            dataset_id=dataset.id
        )

        return JSONResponse(content={"message": "Response Generated Successfully!", "data": results}, status_code=status.HTTP_200_OK)

    finally:
        session.close()

# ✅ 2. File upload endpoint
@app.post("/upload-file/")
async def upload_file(chatbot_id: str, file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            file_path = tmp.name

        if file.filename.endswith(".pdf"):
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
        elif file.filename.endswith(".docx"):
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file.filename.endswith(".txt"):
            from langchain.document_loaders import TextLoader
            loader = TextLoader(file_path)
        else:
            return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

        documents = loader.load()

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        qdrant_obj = QdrantInsertRetrievalAll(
            api_key=os.getenv("QDRANT_API_KEY"),
            url=os.getenv("QDRANT_URL")
        )
        collection_name = f"collection_{chatbot_id}"
        qdrant_obj.insertion(chunks, embeddings, collection_name)

        return {"message": "File uploaded and indexed successfully!"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ✅ 3. Ask question from uploaded file
@app.post("/ask-from-file/")
async def ask_from_file(chatbot_id: str = Form(...), question: str = Form(...)):
    try:
        qdrant_obj = QdrantInsertRetrievalAll(
            api_key=os.getenv("QDRANT_API_KEY"),
            url=os.getenv("QDRANT_URL")
        )
        collection_name = f"collection_{chatbot_id}"
        retriever = qdrant_obj.retrieval(collection_name=collection_name, embeddings=embeddings)

        relevant_docs = retriever.as_retriever().get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in relevant_docs])

        prompt_template = PromptTemplate.from_template(
            "You are an assistant. Use the context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        prompt = prompt_template.format(context=context_text, question=question)
        model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)
        response = model.invoke(prompt)

        return {"message": "Answer generated successfully!", "data": response.content}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ✅ Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006, reload=True)
