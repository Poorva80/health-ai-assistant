import os
import logging
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss"
REPORT_FILE = "search_report.txt"
LOG_FILE = "search_log.log"

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_search(question):
    logging.info(question)

# Custom styling
st.markdown(
    """
    <style>
        .stChatMessageUser {
            background-color: #7886C7;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .stChatMessageBot {
            background-color: #2D336B;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .stSidebar {
            background-color: #A9B5DF;
            padding: 15px;
            border-radius: 10px;
        }
        .stTitle {
            color: #4A90E2;
            text-align: center;
            font-size: 28px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load vector store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Custom prompt template
def set_custom_prompt():
    return PromptTemplate(template="""
        You are a medical assistant. Use the provided context to answer the user's question in a detailed and structured manner.
        If the context does not contain enough information, say "I don't have enough information to answer that question."
        If the question is not related to the medical field, respond with "I am a medical assistant, so I can only answer health-related queries."
        Always provide answers in a clear and concise format, using bullet points or paragraphs where appropriate.

        Context: {context}
    A    Question: {question}

        Answer:
    """, input_variables=["context", "question"])

# Load LLM
def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.1,
        model_kwargs={"token": HF_TOKEN, "max_length": "1024"}
    )

# Format the response
def format_response(response):
    response = response.replace("**", "") 
    response = response.replace("- ", "• ")
    response = response.replace("\n", "<br>")
    return response

# Append search to report
def append_to_report(question):
    with open(REPORT_FILE, "a") as file:
        file.write(question + "\n")
    log_search(question)

# Read search report
def read_report():
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r") as file:
            return file.read()
    return "No searches recorded yet."

# Main function
def main():
    st.markdown("<h1 class='stTitle'>🩺 Medical Chatbot</h1>", unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.markdown("<div class='stSidebar'><h2>🔍 Search History</h2></div>", unsafe_allow_html=True)
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        for i, question in enumerate(st.session_state.search_history, 1):
            st.write(f"{i}. {question}")

        # New Chat button
        if st.button("🆕 New Chat"):
            if 'confirm_new_chat' not in st.session_state:
                st.session_state.confirm_new_chat = True  # Show confirmation dialog
            else:
                st.session_state.confirm_new_chat = False  # Hide confirmation dialog

        if st.session_state.get('confirm_new_chat', False):
            st.warning("Are you sure you want to start a new chat? This will clear the current conversation.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, start new chat"):
                    st.session_state.messages = []  # Reset chat messages
                    st.session_state.confirm_new_chat = False  # Hide confirmation dialog
                    st.rerun()  # Rerun the app to reflect changes
            with col2:
                if st.button("No, cancel"):
                    st.session_state.confirm_new_chat = False  
            
           
        if st.button("🗑️ Clear History"):
            if 'confirm_clear_history' not in st.session_state:
                st.session_state.confirm_clear_history = True  # Show confirmation dialog
            else:
                st.session_state.confirm_clear_history = False  # Hide confirmation dialog

        if st.session_state.get('confirm_clear_history', False):
            st.error("Are you sure you want to clear the history? This will delete all search history and chat logs permanently.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, clear history"):
                    st.session_state.search_history = []  # Clear search history
                    st.session_state.messages = []  # Clear chat messages
                    os.remove(REPORT_FILE) if os.path.exists(REPORT_FILE) else None  # Delete report file
                    os.remove(LOG_FILE) if os.path.exists(LOG_FILE) else None  # Delete log file
                    st.session_state.confirm_clear_history = False  # Hide confirmation dialog
                    st.rerun()  # Rerun
            with col2:
                if st.button("No, cancel"):
                    st.session_state.confirm_clear_history = False  # Hide confirmation dialog        
            
    # Display search report
    if st.sidebar.button("📄 Show Search Report"):
        st.sidebar.text_area("Search Report", read_report(), height=300)

    # Chat history setup
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role_class = 'stChatMessageUser' if msg['role'] == 'user' else 'stChatMessageBot'
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # User input
    prompt = st.chat_input("Ask a medical question...")
    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.session_state.search_history.append(prompt)
        append_to_report(prompt)
        st.markdown(f"<div class='stChatMessageUser'>{prompt}</div>", unsafe_allow_html=True)

        # Load LLM and vector store
        HUGGINGFACE_REPO_ID = "microsoft/BioGPT"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            st.error("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
            return

     
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the vector store.")
            return

        # Show loading spinner while generating the response
        with st.spinner("Generating response..."):
         qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        response = qa_chain.invoke({'query': prompt})
        result = response.get("result", "No response generated.")

        formatted_result = format_response(result)
        st.markdown(f"<div class='stChatMessageBot'>{formatted_result}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'assistant', 'content': formatted_result})

if __name__ == "__main__":
    main()
