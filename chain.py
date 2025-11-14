from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from a_data_ingestion import fetch_youtube_transcript
from b_text_splitter import split_text
from c_embeddings import vector_embeddings
from e_augmentation import augment_fn
from d_similarity_search import retriver_fn
from dotenv import load_dotenv
load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  
llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Thinking",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
    add_to_git_credential=True
)

model = ChatHuggingFace(llm=llm)

print(f"================================== Initializing Chatbot Pipeline =================================")

video_id = input("Enter YouTube Video ID: ").strip()
text = fetch_youtube_transcript(video_id)
chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
vector_store = vector_embeddings(chunks)
print(f"Performing initial similarity search...")
context_text = vector_store.similarity_search("What is DeepMind?", k=3)
context_text = "\n\n".join([doc.page_content for doc in context_text])
print(f" Pipeline initialized successfully!")
print(f"================================== Ready to Answer Questions =================================\n")

conversation_history = []

while True:
    query = input("Enter your question (type 'exit' or 'quit' to stop): ")
    if query.strip().lower() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break
    print(f"\n================================== Processing Your Query =================================")
    print("Retrieving the relevant context for optimal response==================")
    # Format conversation history as a list of dicts for clarity
    formatted_history = "\n".join([
        f"Human: {msg['human']}\nAI: {msg['ai']}" for msg in conversation_history
    ])
    # Retrieve relevant docs for the current query
    parallel_chain = RunnableParallel({
        'context': retriver_fn(vector_store, query, k=3)|RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])),
        'question': RunnablePassthrough()                                                                               
    })
    parser = StrOutputParser()
    def memory_augment(x):
        # Add conversation history to context, formatted as Human/AI messages
        full_context = formatted_history + "\n" + x['context'] if formatted_history else x['context']
        return augment_fn(full_context, x['question'], model)
    main_chain = parallel_chain|RunnableLambda(memory_augment)|model|RunnableLambda(lambda x: parser.parse(x.content))
    final_response = main_chain.invoke(query)
    print(f"\n================================== Final Answer =================================\n")
    print("Final Answer: \n\n", final_response)
    print(f"\n================================== End of Response =================================\n")
    # Store as human/ai message dicts
    conversation_history.append({"human": query, "ai": final_response})