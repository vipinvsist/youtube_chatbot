from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from a_data_ingestion import fetch_youtube_transcript
from b_text_splitter import split_text
from c_embeddings import vector_embeddings
from e_augmentation import augment_fn
from d_similarity_search import retriver_fn
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Thinking",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
    add_to_git_credential=True
)
model = ChatHuggingFace(llm=llm)

video_id = input("Enter YouTube Video ID: ").strip()
query = input("Enter your question: ")

print(f"================================== Initializing Chatbot Pipeline =================================")
text = fetch_youtube_transcript(video_id)
chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
vector_store = vector_embeddings(chunks)
print(f"Performing similarity search for your query...")
context_text = vector_store.similarity_search(query, k=3)
context_text = "\n\n".join([doc.page_content for doc in context_text])
print(f" Context retrieved!")
print(f"================================== Generating Augmented Prompt =================================")
final_prompt = augment_fn(context_text, query, model)
print(f"================================== Generating Answer =================================")
answer = model.invoke(final_prompt)
print(f"\n================================== Final Answer =================================\n")
print("Answer: \n\n", answer.content)
print(f"\n================================== End of Response =================================\n")