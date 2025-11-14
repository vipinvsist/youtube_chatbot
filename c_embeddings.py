from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


def vector_embeddings(chunks):
    """
    Generates vector embeddings for the given text chunks.

    Args:
        chunks (List[str]): A list of text chunks.
        """
    print(f"================================== Creating the Vector Embeddings =================================")
    print(f"Initializing HuggingFace embeddings model: sentence-transformers/all-mpnet-base-v2")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},  # or "cuda"
    encode_kwargs={"normalize_embeddings": True}
)
    print(f" Embeddings model loaded successfully!")
    print(f"Creating FAISS vector store for {len(chunks)} chunks...")
    vector_stores=FAISS.from_documents(embedding=embeddings, documents=chunks)
    print(f" Vector store created successfully!")
    print(f"================================== Vector Embeddings Complete =================================\n")

    return vector_stores



