import streamlit as st
import os

# Import our custom modules
from src.ingestion import ingest_pdf_to_chroma
from src.retrieval import get_retriever
from src.generation import generate_answer

# Set page config for a wider, more professional look
st.set_page_config(page_title="Swiggy AI Assistant", page_icon="🍔", layout="wide")

# Securely load the GitHub token
os.environ["GITHUB_TOKEN"] = os.environ.get("GITHUB_TOKEN", "")

@st.cache_resource(show_spinner=False)
def initialize_database():
    """Initializes and caches the vector database."""
    vectorstore = ingest_pdf_to_chroma("data/Annual-Report-FY-2023-24.pdf")
    retriever = get_retriever(vectorstore)
    return retriever

# --- UI Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/1/12/Swiggy_logo.svg", width=150)
    st.title("System Architecture")
    st.info(
        "This RAG application strictly queries the Swiggy FY 2023-24 Annual Report. "
        "It utilizes ChromaDB for semantic vector search and GPT-4o via GitHub Models "
        "for context-grounded answer generation."
    )
    st.divider()

# --- Main Chat Interface ---
st.title("Swiggy Annual Report Assistant 🍔")
st.markdown("Ask anything about Swiggy's financials, risks, or operations for the 2023-24 fiscal year.")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to answer your questions about the Swiggy Annual Report."}]

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load Database
with st.spinner("Initializing Vector Database..."):
    retriever = initialize_database()

# React to user chat input
if prompt := st.chat_input("E.g., Who is the Group CEO?"):
    
    # 1. Display user message in chat UI
    st.chat_message("user").markdown(prompt)
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching the Annual Report..."):
            response = generate_answer(retriever, prompt)
            
            st.markdown(response["answer"])
            
            # Display sources neatly inside an expander
            with st.expander("View Supporting Context"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**")
                    st.caption(doc.page_content)
                    st.divider()

    # 4. Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})