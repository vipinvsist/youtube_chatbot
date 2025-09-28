from langchain_text_splitters import RecursiveCharacterTextSplitter
from a_data_ingestion import fetch_youtube_transcript

def split_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits the input text into smaller chunks.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # chunks = text_splitter.split_text(text)
    chunks=text_splitter.create_documents([text])
    return chunks                    # f"Number of chunks: {len(chunks)}"

# text=fetch_youtube_transcript("Gfr50f6ZBvo")
# print(split_text(text,chunk_size=1000, chunk_overlap=200))