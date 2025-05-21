import streamlit as st
from Conversational_chain.utils import Conversational_Chain  # Importing your conversation logic from utils.py
from Conversational_chain.conv_ret_db import SessionLocal, ConversationChatHistory  # Import your DB models
from langchain_openai import OpenAIEmbeddings  # Import OpenAI embeddings
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit App setup
st.title("Conversational_Chain-AI Chatbot")

# Input fields
chatbot_id_user = st.text_input("Enter your chatbot ID:")
query_user = st.text_area("Ask a question:")

# Button to submit query
if st.button("Submit"):
    # Open database session
    session = SessionLocal()

    # Check if this chatbot_id exists in the database
    chatbot_id = chatbot_id_user

    # Retrieve last 30 interactions for this user
    conversation_history = session.query(ConversationChatHistory).filter_by(chatbot_id=chatbot_id).order_by(ConversationChatHistory.id.desc()).limit(30).all()

    # Prepare chat history
    chat_history = []
    for chat in conversation_history:
        if chat.query:
            chat_history.append(f"User: {chat.query}")
        if chat.response:
            chat_history.append(f"AI response: {chat.response}")

    # Process the query through the conversational chain
    results = Conversational_Chain(query=query_user, history=chat_history)

    # Save conversation in the database
    new_history = ConversationChatHistory(chatbot_id=chatbot_id, query=query_user, response=results)
    session.add(new_history)
    session.commit()

    # Display AI response
    st.write("AI Response: ", results)

    # Close session
    session.close()
