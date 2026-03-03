import streamlit as st
import os

# Import custom modules
from src.ingestion import ingest_pdf_to_chroma
from src.retrieval import get_retriever
from src.generation import generate_answer

# Set page config for a wider, more professional look
st.set_page_config(page_title="Swiggy AI Assistant", layout="wide")

# Load the GitHub token
os.environ["GITHUB_TOKEN"] = os.environ.get("GITHUB_TOKEN", "")

# Only cache the raw vector database, not the retriever
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    """Initializes and caches the vector database."""
    return ingest_pdf_to_chroma("data/Annual-Report-FY-2023-24.pdf")

# UI Sidebar 
with st.sidebar:
    st.title("System Architecture")
    st.info(
        "This RAG application strictly queries the Swiggy FY 2023-24 Annual Report. "
        "It utilizes ChromaDB for semantic vector search and GPT-4o via GitHub Models "
        "for context-grounded answer generation."
    )
    
    # Add a slider to control the k-value
    st.subheader(" Retrieval Settings")
    k_value = st.slider(
        "Number of Context Chunks (k)", 
        min_value=1, 
        max_value=25, 
        value=10,
        help="Higher values give the AI more context to read, but take longer to process."
    )
    
    st.divider()
    st.caption("Developed for the Newel Engineering Internship Submission.")

# Main Chat Interface 
st.title("Swiggy Annual Report Assistant ")
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
    vectorstore = initialize_vectorstore()

# React to user chat input
if prompt := st.chat_input("E.g., Who is the Group CEO?"):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # Tell the user exactly how many chunks are being searched
        with st.spinner(f"Searching the top {k_value} chunks in the Annual Report..."):
            
            # Create the retriever dynamically using the slider's value
            retriever = get_retriever(vectorstore, k=k_value)
            response = generate_answer(retriever, prompt)
            
            st.markdown(response["answer"])
            
            # Display sources neatly inside an expander
            with st.expander("View Supporting Context"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**")
                    st.caption(doc.page_content)
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})