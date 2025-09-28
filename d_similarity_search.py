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
    retriver=vector_store.as_retriever(search_type='similarity', search_kwargs={'k':k})
    # print(retriver)
    # print(f"Retrieving top {k} similar documents for the query: '{query}'")
    # print(f"Vector Store Type: {type(vector_store)}")
    print(retriver.invoke(query))
    return retriver

# text=fetch_youtube_transcript("Gfr50f6ZBvo")
# chunks=split_text(text,chunk_size=1000, chunk_overlap=200)
# # print(f"Number of chunks: {len(chunks)}")
# vector_store = vector_embeddings(chunks)

# print(retriver_fn(vector_store, "What is deepmind?", k=3)) 

