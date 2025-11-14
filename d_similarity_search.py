from langchain.text_splitter import RecursiveCharacterTextSplitter
from a_data_ingestion import fetch_youtube_transcript
from b_text_splitter import split_text
from c_embeddings import vector_embeddings

def retriver_fn(vector_store, query, k=3):
    """
    Retrieves the top k similar documents from the vector store based on the query.

    Args:
        vector_store: The vector store containing document embeddings.
        query (str): The input query string.
        k (int): The number of top similar documents to retrieve.   
        """
    print(f"================================== Similarity Search in Progress =================================")
    print(f"Query: '{query}'")
    print(f"Retrieving top {k} most relevant documents...")
    retriver=vector_store.as_retriever(search_type='similarity', search_kwargs={'k':k})
    # print(retriver)
    # print(f"Retrieving top {k} similar documents for the query: '{query}'")
    # print(f"Vector Store Type: {type(vector_store)}")
    relevant_docs = retriver.invoke(query)
    print(f" Retrieved {len(relevant_docs)} relevant documents!")
    print(f"================================== Documents Retrieved Successfully =================================\n")
    return retriver

