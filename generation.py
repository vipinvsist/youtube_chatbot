from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from a_data_ingestion import fetch_youtube_transcript
from b_text_splitter import split_text
from c_embeddings import vector_embeddings
from e_augmentation import augment_fn
from d_similarity_search import retriver_fn
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  
query = input("Enter your question: ")               # "Summarize this video in detail."
text=fetch_youtube_transcript("Gfr50f6ZBvo")
chunks=split_text(text,chunk_size=1000, chunk_overlap=200)
# print(f"Number of chunks: {len(chunks)}")
vector_store = vector_embeddings(chunks)
context_text = vector_store.similarity_search(query, k=3)
context_text="\n\n".join([doc.page_content for doc in context_text])
print(100*"--")
final_prompt=augment_fn(context_text, query)
answer = llm.invoke(final_prompt)


print("Answer: \n\n", answer.content)