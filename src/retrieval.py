def get_retriever(vectorstore, k: int):
    """Creates a retriever from the vector store to fetch top k chunks."""
    return vectorstore.as_retriever(search_kwargs={"k": k})