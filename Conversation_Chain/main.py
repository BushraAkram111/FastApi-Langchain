# Import FastAPI framework
from fastapi import FastAPI, File, UploadFile
from utils import QdrantInsertRetrievalAll, Conversational_Chain # Import custom utilities including your LLM chain and Qdrant logic
import tempfile
import os
from langchain_openai import OpenAIEmbeddings # Import OpenAI embeddings via LangChain
import uvicorn
from pydantic import BaseModel # For data validation
from conv_ret_db import SessionLocal, ConversationChatHistory # Import database session and chat history table
import tempfile , random , datetime, string # Miscellaneous utilities
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from operator import itemgetter # For dynamic dictionary access
import ast
from fastapi.responses import JSONResponse # For sending structured JSON responses
from fastapi import status
from dotenv import load_dotenv 
load_dotenv()

# Get your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the embeddings model to use
Embeddings_model = "text-embedding-3-small"

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(model = Embeddings_model, api_key = OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Define the input data model expected in the request
class QueryRequest(BaseModel):
    chatbot_id: str
    query: str

# Define your POST endpoint
@app.post("/Conversation")
async def Convo_chain(request: QueryRequest):
    # Extract chatbot ID and query from the request
    chatbot_id_user = request.chatbot_id
    query_user = request.query
    print("query_user: ", type(query_user))
    print("Received chatbot_id_user:", chatbot_id_user)

    # Start a database session
    session = SessionLocal()
    try:
        # Check if this chatbot ID already exists in the database
        chatbot_id_table = session.query(ConversationChatHistory).filter_by(chatbot_id=chatbot_id_user).first()
        print(chatbot_id_table)
        
        if chatbot_id_table:
            # If yes, use the same ID
            chatbot_id = chatbot_id_table.chatbot_id
            print(f"Existing chatbot_id found: {chatbot_id}")
        else:
            # Otherwise, assign it as a new ID
            chatbot_id = chatbot_id_user
            print(f"Generated new chatbot_id: {chatbot_id}")
        
        # Retrieve last 30 interactions for this user
        conversation_history = session.query(ConversationChatHistory).filter_by(chatbot_id=chatbot_id).order_by(ConversationChatHistory.id.desc()).limit(30).all()

        # Prepare history list to pass into the prompt
        chat_history = []
        for chat in conversation_history:
            if chat.query:
                chat_history.append(f"User: {chat.query}")
            if chat.response:
                chat_history.append(f"AI response: {chat.response}")

        # Initialize the OpenAI chat model via LangChain
        llm_model = ChatOpenAI(model='gpt-4o-mini', openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

        # Greeting detection prompt
        greeting_prompt = """
            You are a smart AI assistant. Your task is to classify whether the given user query is a greeting (like 'Hi', 'Hello', etc.).
            If the query is a greeting, respond with: ["yes", "your greeting response"]
            If the query is not a greeting, respond with: ["no", "statusCode:404"]

            User query: {query}
            Make sure to follow the exact output format: ["yes/no", "greeting_response"]
            """

        # Format the prompt with LangChain
        greeting_prompt = ChatPromptTemplate.from_template(greeting_prompt)

        # Prepare the input dictionary
        query_input = {"query": query_user}

        # Run the greeting chain (to check if it's a greeting)
        greeting_chain = ({"query": itemgetter("query")}  | greeting_prompt | llm_model)
        greeting_response = greeting_chain.invoke(query_input)

        # Convert model's response from string to list
        response = ast.literal_eval(greeting_response.content)
        label = response[0]

        # If it's a greeting, save and return the greeting message
        if label.lower().endswith("yes"):
            message = response[1]
            new_history = ConversationChatHistory(
                chatbot_id=chatbot_id,
                query=query_user,
                response=message
            )
            session.add(new_history)
            session.commit()
            return JSONResponse(content={"message": "Response Generated Successfully!", "data": message}, status_code=status.HTTP_200_OK)

        # If it's not a greeting, send query + history to the main AI chain
        results = Conversational_Chain(query = query_user, history = chat_history)

        # Save the interaction to the database
        new_history = ConversationChatHistory(
            chatbot_id=chatbot_id,
            query=query_user,
            response=results
        )
        session.add(new_history)
        session.commit()

        # Return the response
        return JSONResponse(content={"message": "Response Generated Successfully!", "data": results}, status_code=status.HTTP_200_OK)

    finally:
        # Close the DB session no matter what
        session.close()

# If running the file directly, launch Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006, reload=True)
