import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()

    # Set up OpenAI
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Simple page config
    st.set_page_config(page_title="AI Chat", page_icon="💬")

    # Initialize chat model
    llm = ChatOpenAI(model_name="gpt-4")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar with settings
    with st.sidebar:
        st.title("Settings")
        model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])
        temperature = st.slider("Response creativity", 0.0, 1.0, 0.7)

    # Main chat interface
    st.title("QA AI Chat")
    st.caption("Ask me anything and I'll try to help!")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Update model with selected settings
                    llm = ChatOpenAI(model_name=model, temperature=temperature)
                    response = llm.invoke(prompt)
                    ai_response = response.content
                    
                    # Display and store AI response
                    st.write(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
    # streamlit run QA_chat_app.py