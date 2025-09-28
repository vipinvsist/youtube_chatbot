from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from a_data_ingestion import fetch_youtube_transcript
from b_text_splitter import split_text
from dotenv import load_dotenv
load_dotenv()


def vector_embeddings(chunks):
    """
    Generates vector embeddings for the given text chunks.

    Args:
        chunks (List[str]): A list of text chunks.
        """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_stores=FAISS.from_documents(embedding=embeddings, documents=chunks)

    return vector_stores

text=fetch_youtube_transcript("Gfr50f6ZBvo")
chunks=split_text(text,chunk_size=1000, chunk_overlap=200)
print(f"Number of chunks: {len(chunks)}")
# print(vector_embeddings(chunks))

vector_store = vector_embeddings(chunks)

# # Get all document IDs
# all_doc_ids = list(vector_store.index_to_docstore_id.values())

# # Extract all documents
# all_documents = []
# for doc_id in all_doc_ids:
#     document = vector_store.docstore.search(doc_id)
#     all_documents.append(document)
    
# print(f"Retrieved {len(all_documents)} documents")
# for i, doc in enumerate(all_documents[:3]):  # Show first 3
#     print(f"Document {i}: {doc.page_content[:100]}...")
# # print(vector_embeddings(chunks).index_to_docstore_id)
# # print(vector_embeddings(chunks).get_by_ids(['648a5a7a-7f26-4795-857b-15e620d5f6ec']))