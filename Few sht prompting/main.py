from fastapi import FastAPI, File, UploadFile, Form
from utils import QdrantInsertRetrievalAll, Conversational_Chain
import tempfile
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import uvicorn
from pydantic import BaseModel
from conv_ret_db import SessionLocal, ConversationChatHistory
from langchain_core.prompts.chat import ChatPromptTemplate
from operator import itemgetter
import ast
from fastapi.responses import JSONResponse
from fastapi import status
from dotenv import load_dotenv
from langsmith import Client
from dotenv import find_dotenv
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

client = Client()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
Embeddings_model = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=Embeddings_model, api_key=OPENAI_API_KEY)

app = FastAPI()

class QueryRequest(BaseModel):
    chatbot_id: str
    query: str

# Define retrieve_from_qdrant to interact with the Qdrant client and retrieve documents
def retrieve_from_qdrant(query: str, collection_name: str):
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance

    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

    # Perform vector search (ensure you have a vector to search with)
    query_vector = embeddings.encode([query])  # Assuming you use embeddings to search
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector[0],  # Assuming embeddings returns a list of vectors
        top=5  # Number of top results to retrieve
    )
    return results

@app.post("/Conversation")
async def Convo_chain(request: QueryRequest):
    chatbot_id = request.chatbot_id
    query_user = request.query
    session = SessionLocal()

    try:
        conversation_history = session.query(ConversationChatHistory).filter_by(chatbot_id=chatbot_id).order_by(ConversationChatHistory.id.desc()).limit(30).all()
        chat_history = [f"User: {chat.query}" if chat.query else f"AI response: {chat.response}" for chat in conversation_history]

        query_count = session.query(ConversationChatHistory).filter_by(chatbot_id=chatbot_id).count()
        batch_number = (query_count // 20) + 1
        dataset_name = f"conversation_batch_{batch_number}"
        existing_datasets = client.list_datasets()
        dataset = next((d for d in existing_datasets if d.name == dataset_name), None)
        if not dataset:
            dataset = client.create_dataset(dataset_name)

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
            client.create_example(inputs={"query": query_user}, outputs={"response": message}, dataset_id=dataset.id)
            return JSONResponse(content={"message": "Response Generated Successfully!", "data": message}, status_code=status.HTTP_200_OK)

        # ✅ Few-shot prompting to extract clean query
        examples = [
            {"input": "Hello, my name is John. What is AI? today weather is hot", "output": "What is AI?"},
            {"input": "Hi! Just testing. What's machine learning?", "output": "What's machine learning?"},
            {"input": "Greetings! How does LangChain work?", "output": "How does LangChain work?"}
        ]

        example_prompt = PromptTemplate(input_variables=["input", "output"], template="Input: {input}\nOutput: {output}")
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Extract the main question from user's input. Ignore greetings or irrelevant chatter.",
            suffix="Input: {user_input}\nOutput:",
            input_variables=["user_input"]
        )

        extract_chain = LLMChain(llm=llm_model, prompt=few_shot_prompt)
        main_query = extract_chain.run({"user_input": query_user})

        results = Conversational_Chain(query=main_query, history=chat_history)

        session.add(ConversationChatHistory(chatbot_id=chatbot_id, query=main_query, response=results))
        session.commit()
        client.create_example(inputs={"query": main_query}, outputs={"response": results}, dataset_id=dataset.id)

        return JSONResponse(content={"message": "Response Generated Successfully!", "data": results}, status_code=status.HTTP_200_OK)

    except Exception as e:
        return JSONResponse(content={"error": f"Server error: {str(e)}"}, status_code=500)

    finally:
        session.close()

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

        qdrant_obj = QdrantInsertRetrievalAll(api_key=os.getenv("QDRANT_API_KEY"), url=os.getenv("QDRANT_URL"))
        collection_name = f"collection_{chatbot_id}"
        qdrant_obj.insertion(chunks, embeddings, collection_name)

        return {"message": "File uploaded and indexed successfully! You can now query the document."}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/query-uploaded-file/")
async def query_uploaded_file(chatbot_id: str, query: str):
    try:
        from qdrant_client import QdrantClient
        from langsmith import Client
        from langchain_core.tracers.context import tracing_v2_enabled

        # Initialize LangSmith client
        langsmith_client = Client()

        # Create dataset name based on chatbot_id
        dataset_name = f"document_retrieval_{chatbot_id}"
        existing_datasets = langsmith_client.list_datasets()
        dataset = next((d for d in existing_datasets if d.name == dataset_name), None)
        if not dataset:
            dataset = langsmith_client.create_dataset(dataset_name)

        with tracing_v2_enabled():
            # Initialize LLM for few-shot prompting
            llm_model = ChatOpenAI(
                model='gpt-3.5-turbo',
                openai_api_key=OPENAI_API_KEY,
                temperature=0,
                verbose=False
            )

            # ✅ Few-shot prompting to extract clean query (same as in conversation endpoint)
            examples = [
                {"input": "Hello, my name is John. What is AI? today weather is hot", "output": "What is AI?"},
                {"input": "Hi! Just testing. What's machine learning?", "output": "What's machine learning?"},
                {"input": "Greetings! How does LangChain work?", "output": "How does LangChain work?"},
                {"input": "The weather is nice. Can you explain neural networks?", "output": "Explain neural networks"},
                {"input": "Good morning! I need help with pandas dataframe", "output": "Help with pandas dataframe"}
            ]

            example_prompt = PromptTemplate(
                input_variables=["input", "output"],
                template="Input: {input}\nOutput: {output}"
            )
            
            few_shot_prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix="Extract the main question/request from user's input. Ignore greetings or irrelevant chatter.",
                suffix="Input: {user_input}\nOutput:",
                input_variables=["user_input"]
            )

            extract_chain = LLMChain(llm=llm_model, prompt=few_shot_prompt)
            main_query = extract_chain.invoke(
                {"user_input": query},
                config={"run_name": "QueryExtraction", "tags": ["few-shot", chatbot_id]}
            )["text"]

            # Initialize Qdrant client
            qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=30
            )
            
            collection_name = f"collection_{chatbot_id}"
            
            # Verify collection exists
            try:
                collection_info = qdrant_client.get_collection(collection_name)
                if not collection_info:
                    return JSONResponse(
                        content={"error": f"Collection '{collection_name}' not found. Please upload documents first."},
                        status_code=404
                    )
            except Exception as e:
                return JSONResponse(
                    content={"error": f"Failed to verify collection: {str(e)}"},
                    status_code=404
                )

            # Generate embeddings for the extracted query
            query_embedding = embeddings.embed_query(main_query)

            # Perform search
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            # Process results
            formatted_results = []
            for hit in search_results:
                content = hit.payload.get("page_content", "")
                # Clean content
                cleaned_content = " ".join(content.replace("-\n", "").split())
                formatted_results.append({
                    "id": str(hit.id),
                    "score": round(hit.score, 4),
                    "content": cleaned_content[:300] + ("..." if len(cleaned_content) > 300 else ""),
                    "source": hit.payload.get("metadata", {}).get("source", "unknown"),
                    "page": hit.payload.get("metadata", {}).get("page", 1)
                })

            # Generate summary using LLM
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Summarize the following document excerpts in response to the user's original query."),
                ("human", "Original query: {query}\nExtracted query: {main_query}\n\nDocument excerpts:\n{results}")
            ])
            
            summary_chain = summary_prompt | llm_model
            summary = summary_chain.invoke({
                "query": query,
                "main_query": main_query,
                "results": "\n\n".join([f"Document {i+1} (Relevance: {res['score']:.2f}):\n{res['content']}" 
                                      for i, res in enumerate(formatted_results)])
            }).content

            # Log to LangSmith
            langsmith_client.create_example(
                inputs={
                    "original_query": query,
                    "extracted_query": main_query,
                    "chatbot_id": chatbot_id
                },
                outputs={
                    "summary": summary,
                    "results": formatted_results,
                    "num_results": len(formatted_results)
                },
                dataset_id=dataset.id,
                metadata={
                    "retrieval_method": "vector_search",
                    "collection": collection_name
                }
            )

            return {
                "query": query,
                "extracted_query": main_query,
                "summary": summary,
                "results": formatted_results,
                "status": "success",
                "langsmith_trace": f"https://app.langchain.com/trace/{langsmith_client.last_run_id}" if hasattr(langsmith_client, 'last_run_id') else None
            }

    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006, reload=True)
