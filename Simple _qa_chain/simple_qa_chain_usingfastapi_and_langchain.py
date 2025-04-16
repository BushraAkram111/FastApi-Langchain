from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os, uvicorn

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    text: str


llm = ChatOpenAI(model_name="gpt-4o-mini") 

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    """Takes user input, sends it to OpenAI, and returns a response."""
    question = request.text
    response = llm.invoke(question)
    return {"response": response.content}

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)