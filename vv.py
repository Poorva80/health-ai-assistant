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
        /* General body styling */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f0f2f6;
        }

        /* Title styling */
        .stTitle {
            color: #4A90E2;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
            animation: fadeIn 2s ease-in-out;
        }

        /* Chat message styling */
        .stChatMessageUser {
            background-color: #7886C7;
            color: white;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 15px;
            max-width: 70%;
            margin-left: auto;
            animation: slideInRight 0.5s ease-in-out;
        }

        .stChatMessageBot {
            background-color: #2D336B;
            color: white;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 15px;
            max-width: 70%;
            margin-right: auto;
            animation: slideInLeft 0.5s ease-in-out;
        }

        /* Sidebar styling */
        .stSidebar {
            background-color: #A9B5DF;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            animation: fadeIn 1s ease-in-out;
        }

        .stSidebar h2 {
            color: #2D336B;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
        }

        /* Button styling */
        .stButton button {
            background-color: #4A90E2;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }

        .stButton button:hover {
            background-color: #2D336B;
            transform: scale(1.05);
        }

        /* Input box styling */
        .stTextInput input {
            border: 2px solid #4A90E2;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }

        .stTextInput input:focus {
            border-color: #2D336B;
            outline: none;
        }

        /* Animation keyframes */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideInLeft {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Hover effects for chat messages */
        .stChatMessageUser:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transform: translateY(-5px);
            transition: box-shadow 0.3s ease, transform 0.3s ease;
        }

        .stChatMessageBot:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transform: translateY(-5px);
            transition: box-shadow 0.3s ease, transform 0.3s ease;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f0f2f6;
        }

        ::-webkit-scrollbar-thumb {
            background: #4A90E2;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #2D336B;
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
        You are a medical chatbot designed to provide detailed and accurate answers to health-related questions. 
        Your task is to answer questions about diseases, symptoms, treatments, medications, and other medical topics. 
        If the question is not related to health or medicine, respond with: 
        "I am a medical chatbot and can only assist with health-related questions. Please ask a question about health or medicine."

        **Instructions for Medical Questions:**
        1. Provide a detailed and accurate answer based on the context.
        2. Use bullet points, paragraphs, and clear headings for better readability.
        3. Include all relevant information from the context.
        4. If the context does not contain enough information, state: "I don't have enough information to answer this question. Please consult a healthcare professional."

        Context: {context}
        Question: {question}

        Answer:
    """, input_variables=["context", "question"])

# Load LLM
def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.1,
        repetition_penalty=1.2,
        model_kwargs={"token": HF_TOKEN, "max_length": "1024"}
    )

# Format the response
def format_response(response):
    # Replace markdown-style bullet points with HTML bullet points
    response = response.replace("**", "") 
    response = response.replace("- ", "• ")  # Convert dashes to bullets
    response = response.replace("\n", "<br>")  # Add line breaks for paragraphs
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

# Check if the question is health-related
def is_health_related(question):
    health_keywords = [
        "health", "medical", "disease", "symptom", "treatment", "medicine", 
        "doctor", "hospital", "diagnosis", "therapy", "paralysis", "pain", 
        "fever", "cough", "infection", "surgery", "vaccine", "nutrition",
        "headache", "diabetes", "cancer", "heart", "blood", "pressure", 
        "allergy", "asthma", "arthritis", "depression", "anxiety", "mental",
        "fitness", "exercise", "diet", "weight", "obesity", "pregnancy",
        "childbirth", "pediatric", "elderly", "senior", "chronic", "acute",
        "injury", "wound", "burn", "fracture", "sprain", "stroke", "attack",
        "covid", "virus", "bacteria", "antibiotic", "pharmacy", "prescription",
        "allergic", "rash", "itch", "swelling", "nausea", "vomit", "diarrhea",
        "constipation", "ulcer", "acne", "eczema", "psoriasis", "osteoporosis",
        "thyroid", "hormone", "insulin", "cholesterol", "kidney", "liver",
        "lung", "respiratory", "asthma", "pneumonia", "tuberculosis", "malaria",
        "dengue", "flu", "influenza", "head injury", "concussion", "migraine",
        "seizure", "epilepsy", "autism", "adhd", "alzheimer", "dementia",
        "parkinson", "multiple sclerosis", "lupus", "hiv", "aids", "hepatitis",
        "std", "sti", "sexual health", "reproductive", "contraception", "fertility",
        "menopause", "period", "menstrual", "pms", "endometriosis", "pcos",
        "prostate", "testicular", "breast", "ovarian", "cervical", "colon",
        "rectal", "bladder", "urinary", "incontinence", "prolapse", "hernia",
        "appendicitis", "gallbladder", "pancreas", "spleen", "bone", "muscle",
        "joint", "ligament", "tendon", "cartilage", "spine", "back pain", "neck pain",
        "shoulder pain", "knee pain", "hip pain", "foot pain", "hand pain", "wrist pain",
        "elbow pain", "ankle pain", "toothache", "gum", "dental", "oral", "vision",
        "eye", "glaucoma", "cataract", "hearing", "ear", "nose", "throat", "sinus",
        "tonsil", "adenoid", "sleep", "insomnia", "apnea", "snoring", "fatigue",
        "weakness", "dizziness", "vertigo", "fainting", "coma", "sepsis", "shock",
        "bleeding", "clot", "embolism", "aneurysm", "varicose", "hemorrhoid",
        "anemia", "leukemia", "lymphoma", "myeloma", "sarcoma", "carcinoma",
        "melanoma", "tumor", "cyst", "polyp", "ulcer", "abscess", "boil", "cellulitis",
        "gangrene", "frostbite", "heatstroke", "dehydration", "malnutrition",
        "vitamin", "mineral", "calcium", "iron", "potassium", "sodium", "magnesium",
        "zinc", "protein", "carbohydrate", "fat", "fiber", "water", "hydration",
        "detox", "cleanse", "fasting", "vegan", "vegetarian", "keto", "paleo",
        "mediterranean", "dash", "gluten", "lactose", "allergy", "intolerance",
        "sensitivity", "celiac", "crohn", "colitis", "ibs", "ibd", "gerd", "acid reflux",
        "heartburn", "indigestion", "bloating", "gas", "flatulence", "belching",
        "hiccups", "jaundice", "cirrhosis", "pancreatitis", "diverticulitis",
        "appendicitis", "peritonitis", "meningitis", "encephalitis", "myocarditis",
        "endocarditis", "pericarditis", "pneumonitis", "bronchitis", "pharyngitis",
        "laryngitis", "sinusitis", "otitis", "conjunctivitis", "urethritis", "cystitis",
        "pyelonephritis", "glomerulonephritis", "nephritis", "nephrosis", "nephropathy",
        "urolithiasis", "renal", "kidney stone", "gallstone", "pancreatic stone",
        "salivary stone", "tonsil stone", "bladder stone", "prostate stone",
        "vaginal", "penile", "scrotal", "testicular", "ovarian", "uterine", "cervical",
        "vulvar", "clitoral", "labial", "breast", "nipple", "areola", "mammary",
        "prostate", "penile", "testicular", "scrotal", "epididymal", "vas deferens",
        "seminal", "sperm", "ejaculation", "erection", "impotence", "libido", "orgasm",
        "fertility", "infertility", "sterility", "contraception", "condom", "pill",
        "iud", "implant", "injection", "patch", "ring", "diaphragm", "spermicide",
        "vasectomy", "tubal ligation", "abortion", "miscarriage", "stillbirth",
        "ectopic", "placenta", "amniotic", "umbilical", "cord", "fetal", "neonatal",
        "infant", "toddler", "child", "adolescent", "teen", "adult", "elderly", "senior"
    ]
    is_health = any(keyword in question.lower() for keyword in health_keywords)
    logging.info(f"Question: {question} | Is Health-Related: {is_health}")
    return is_health
# Main function
def main():
    st.markdown("<h1 class='stTitle'>🩺 Healthcare Chatbot</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='animate__animated animate__fadeIn' style='text-align: center; font-size: 18px; color: #4A90E2;'>Please enter your question in CAPITAL WORDS.</p>",
        unsafe_allow_html=True
    )

    # Sidebar navigation
    with st.sidebar:
        st.markdown("<div class='stSidebar'><h2>🔍 Search History</h2></div>", unsafe_allow_html=True)
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        for i, question in enumerate(st.session_state.search_history, 1):
            st.write(f"{i}. {question}")
        
        # New Chat button
        if st.button("🆕 New Chat"):
            st.session_state.messages = []  # Reset chat messages
            st.rerun()  

        # Clear History button
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            os.remove(REPORT_FILE) if os.path.exists(REPORT_FILE) else None
            os.remove(LOG_FILE) if os.path.exists(LOG_FILE) else None
            st.experimental_rerun()

        # Display search report
        if st.button("📄 Show Search Report"):
            st.text_area("Search Report", read_report(), height=300)

    # Chat history setup
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role_class = 'stChatMessageUser' if msg['role'] == 'user' else 'stChatMessageBot'
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # User input
    prompt = st.chat_input("Ask a medical question...")
    if prompt:
        # Check if the question is health-related
        if not is_health_related(prompt):
            st.markdown(f"<div class='stChatMessageBot'>I am a medical chatbot and can only assist with health-related questions. Please ask a question about health or medicine.</div>", unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': "I am a medical chatbot and can only assist with health-related questions. Please ask a question about health or medications."})
            return  # Exit early if the question is not health-related

        # If the question is health-related, proceed
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

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 10, 'fetch_k': 20}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        # Generate response
        response = qa_chain.invoke({'query': prompt})
        result = response.get("result", "No response generated.")

        # Format the response
        formatted_result = format_response(result)
        st.markdown(f"<div class='stChatMessageBot'>{formatted_result}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'assistant', 'content': formatted_result})

if __name__ == "__main__":
    main()