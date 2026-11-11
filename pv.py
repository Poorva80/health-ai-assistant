import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
import re

# Load environment variables
load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss"

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
        You are a medical AI assistant. You **must only** answer medical-related questions.
        - If the question is not related to medicine, health, diseases, treatments, or medical research, say: "I can only assist with medical-related queries."
        - Do not answer anything about technology, AI, computers, programming, or any other unrelated field.
        - Use the given context to provide factual medical information.
        - If the context does not contain relevant information, state that clearly.
        - Format your response using paragraphs and bullet points for clarity.
        
        Context: {context}
        Question: {question}
    """, input_variables=["context", "question"])

# Function to check if a question is medical-related
def is_medical_question(question):
    medical_keywords = ["doctor", "disease", "health", "symptom", "treatment", "medicine", "surgery", "virus", "infection", "therapy", "diagnosis"]
    
    # Check if the question contains at least one medical keyword
    return any(re.search(rf"\b{kw}\b", question, re.IGNORECASE) for kw in medical_keywords)

# Load LLM
def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.1,
        repetition_penalty=1.2,  # Explicitly passing repetition_penalty
        model_kwargs={"token": HF_TOKEN, "max_length": 1024}
    )

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
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.experimental_rerun()

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
        st.markdown(f"<div class='stChatMessageUser'>{prompt}</div>", unsafe_allow_html=True)

        # Restrict to medical questions only
        if not is_medical_question(prompt):
            st.markdown("<div class='stChatMessageBot'>I can only assist with medical-related queries.</div>", unsafe_allow_html=True)
            return

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

        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        response = qa_chain.invoke({'query': prompt})
        result = response.get("result", "No response generated.")
        st.markdown(f"<div class='stChatMessageBot'>{result}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'assistant', 'content': result})

if __name__ == "__main__":
    main()
