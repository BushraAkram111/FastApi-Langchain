from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os, uvicorn

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()
llm = ChatOpenAI(model_name="gpt-4o-mini")

# BaseModel for asking and summarizing (no default language needed)
class TextRequest(BaseModel):
    text: str

# BaseModel for translation with default language = English
class TranslationRequest(BaseModel):
    text: str
    language: str = "English"
#Ask question
@app.post("/ask/")
async def ask_question(request: TextRequest):
    question = request.text
    response = llm.invoke(question)
    return {"response": response.content}
#Translate language
@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    prompt = f"Translate the following text to {request.language}: {request.text}"
    response = llm.invoke(prompt)
    return {"translated_text": response.content, "language": request.language}
#Summarize text
@app.post("/summarize/")
async def summarize_text(request: TextRequest):
    prompt = f"Summarize the following text: {request.text}"
    response = llm.invoke(prompt)
    return {"summary": response.content}

# Run with: uvicorn app:app --reload
