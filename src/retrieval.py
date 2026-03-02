def get_retriever(vectorstore):
    """Creates a retriever from the vector store to fetch top 10 chunks."""
    return vectorstore.as_retriever(search_kwargs={"k": 10})