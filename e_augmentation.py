from multiprocessing import context
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from a_data_ingestion import fetch_youtube_transcript
from b_text_splitter import split_text
from c_embeddings import vector_embeddings
from dotenv import load_dotenv
load_dotenv()

def augment_fn(context, question):
    """
    Augments the input question based on the provided context using a language model.

    Args:
        context (str): The context to be used for augmentation.
        question (str): The input question to be augmented.

    Returns:
        str: The augmented question.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
    final_prompt = prompt.invoke({"context": context, "question": question})
    return final_prompt

# text=fetch_youtube_transcript("Gfr50f6ZBvo")
# chunks=split_text(text,chunk_size=1000, chunk_overlap=200)
# # print(f"Number of chunks: {len(chunks)}")
# vector_store = vector_embeddings(chunks)
# context_text = vector_store.similarity_search("What is DeepMind?", k=3)
# context_text="\n\n".join([doc.page_content for doc in context_text])
# print(100*"--")

# print(augment_fn(context_text, "What is DeepMind?"))