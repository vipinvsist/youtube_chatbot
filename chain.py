from html import parser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from a_data_ingestion import fetch_youtube_transcript
from b_text_splitter import split_text
from c_embeddings import vector_embeddings
from e_augmentation import augment_fn
from d_similarity_search import retriver_fn
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  

text=fetch_youtube_transcript("Gfr50f6ZBvo")
chunks=split_text(text,chunk_size=1000, chunk_overlap=200)
# print(f"Number of chunks: {len(chunks)}")
vector_store = vector_embeddings(chunks)
context_text = vector_store.similarity_search("What is DeepMind?", k=3)
context_text="\n\n".join([doc.page_content for doc in context_text])
print(100*"--")

query=input("Enter your question: ")

parallel_chain = RunnableParallel({
    'context': retriver_fn(vector_store,query, k=3)|RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])),
    'question': RunnablePassthrough()                                                                               
})

# response = parallel_chain.invoke("What is Demis?")
# print("Parallel Chain Response: \n\n", response)

parser = StrOutputParser()
main_chain = parallel_chain|RunnableLambda(lambda x: augment_fn(x['context'], x['question']))|llm|RunnableLambda(lambda x: parser.parse(x.content))
final_response = main_chain.invoke(query)
print("Final Answer: \n\n", final_response)