from multiprocessing import context
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from a_data_ingestion import fetch_youtube_transcript
from b_text_splitter import split_text
from c_embeddings import vector_embeddings
from dotenv import load_dotenv
load_dotenv()

def augment_fn(context, question, llm):
    """
    Augments the input question based on the provided context using a language model.

    Args:
        context (str): The context to be used for augmentation.
        question (str): The input question to be augmented.
        llm: The language model instance (e.g., ChatHuggingFace)

    Returns:
        str: The augmented question.
    """
    print(f"================================== Augmenting Prompt =================================")
    print(f"Processing question with context...")
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
    print(f" Prompt augmentation complete!")
    print(f"================================== Augmentation Complete =================================\n")
    return final_prompt