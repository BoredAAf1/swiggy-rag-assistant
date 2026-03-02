# Swiggy Annual Report AI Assistant 

A high-performance Retrieval-Augmented Generation (RAG) application designed to extract and analyze data from the Swiggy FY 2023-24 Annual Report. 

## Features
- **Modular Architecture:** Clean separation of concerns (Ingestion, Retrieval, Generation).
- **Advanced LLM:** Powered by **GPT-4o** via GitHub Models for high-reasoning accuracy.
- **Vector Search:** Uses **ChromaDB** and HuggingFace embeddings for semantic document retrieval.
- **Anti-Hallucination:** Strict system prompting to ensure answers are grounded only in the provided PDF.

## Project Structure
```text
├── data/           # Source PDF (Swiggy Annual Report)
├── src/            
│   ├── ingestion.py   # PDF processing & vector storage logic
│   ├── retrieval.py   # Similarity search configuration
│   └── generation.py  # GPT-4o API integration & prompting
├── app.py          # Streamlit UI Layer
└── README.md
