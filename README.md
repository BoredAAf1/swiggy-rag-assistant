# Swiggy Annual Report AI Assistant 

A Retrieval-Augmented Generation (RAG) application designed to extract and analyze data from the Swiggy FY 2023-24 Annual Report. 
Source link:-
https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf

## Features
- **Modular Architecture:** Clean separation of concerns (Ingestion, Retrieval, Generation).
- **Advanced LLM:** Powered by **GPT-4o** via GitHub Models for high-reasoning accuracy.
- **Vector Search:** Uses **ChromaDB** and HuggingFace embeddings for semantic document retrieval.
- **Anti-Hallucination:** Strict system prompting to ensure answers are grounded only in the provided PDF.

## Project Structure
```text
├── data/           
├── src/            
│   ├── ingestion.py   
│   ├── retrieval.py  
│   └── generation.py  
├── app.py          
└── README.md

