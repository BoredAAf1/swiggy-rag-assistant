import os
from openai import OpenAI

def generate_answer(retriever, query: str):
    """
    Retrieves context from the vector database and calls the GitHub Models API (GPT-4o) 
    directly without using LangChain's chain abstractions.
    """
    # 1. Manually retrieve the relevant chunks from Chroma
    retrieved_docs = retriever.invoke(query)
    
    # 2. Format the context into a single clean string
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 3. Setup the OpenAI client to point to the free GitHub Models server
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ.get("GITHUB_TOKEN")
    )
    
    # 4. Construct the prompt strictly enforcing no hallucinations
    system_prompt = (
        "You are an AI assistant built to answer questions based strictly on the Swiggy Annual Report. "
        "Use the following pieces of retrieved context to answer the question. "
        "Please provide a summary if the answer is long"
        "If you don't know the answer or if the answer is not in the context, just say that you don't know. "
        "Do not hallucinate or use outside knowledge.\n\n"
        f"Context:\n{context_text}"
    )

    # 5. Call the API directly
    response = client.chat.completions.create(
        model="gpt-4o", # Using GPT-4o via GitHub Models
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    
    # Return both the generated text and the documents used (so the UI can still show sources)
    return {
        "answer": response.choices[0].message.content,
        "context": retrieved_docs
    }