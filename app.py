import streamlit as st
import os

# Import our custom modules
from src.ingestion import ingest_pdf_to_chroma
from src.retrieval import get_retriever
from src.generation import generate_answer

# Set your GitHub Token here
os.environ["GITHUB_TOKEN"] = os.environ.get("GITHUB_TOKEN", "")

st.title("Swiggy Annual Report Q&A Assistant 🍔")
st.write("Ask any question about the Swiggy Annual Report, and I will answer strictly based on the document.")

# Cache the database setup so the PDF is only processed once
@st.cache_resource
def initialize_database():
    vectorstore = ingest_pdf_to_chroma("data/Annual-Report-FY-2023-24.pdf")
    retriever = get_retriever(vectorstore)
    return retriever

# Start the database
retriever = initialize_database()

# User Interface
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching the Annual Report with GPT-4o..."):
        # We now call our custom generation function instead of a LangChain chain
        response = generate_answer(retriever, query)
        
        st.write("### Answer")
        st.write(response["answer"])
        
        with st.expander("View Supporting Context"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Chunk {i+1} (From Page {doc.metadata.get('page', 'Unknown')}):**")
                st.write(doc.page_content)
                st.divider()