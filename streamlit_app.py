import streamlit as st
from a_data_ingestion import fetch_youtube_transcript
from b_text_splitter import split_text
from c_embeddings import vector_embeddings
from d_similarity_search import retriver_fn
from e_augmentation import augment_fn
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

st.set_page_config(page_title="YouTube Chatbot", layout="wide")
st.title("🎬 YouTube Video Chatbot")

# Sidebar for video ID input
video_id = st.sidebar.text_input(
    "Enter YouTube Video ID:",
    value=st.session_state.get("video_id", ""),
    placeholder="e.g. U8J32Z3qV8s"  
)

# Only reload transcript/vector store if video_id changes
if video_id:
    st.sidebar.markdown(f"[Watch Video](https://www.youtube.com/watch?v={video_id})", unsafe_allow_html=True)
    st.video(f"https://www.youtube.com/watch?v={video_id}")

    if (
        "video_id" not in st.session_state or
        st.session_state.video_id != video_id or
        "vector_store" not in st.session_state
    ):
        with st.spinner("Fetching transcript and preparing chatbot..."):
            transcript = fetch_youtube_transcript(video_id)
            chunks = split_text(transcript, chunk_size=1000, chunk_overlap=200)
            vector_store = vector_embeddings(chunks)
            st.session_state.transcript = transcript
            st.session_state.chunks = chunks
            st.session_state.vector_store = vector_store
            st.session_state.video_id = video_id
    else:
        transcript = st.session_state.transcript
        chunks = st.session_state.chunks
        vector_store = st.session_state.vector_store

    # HuggingFace model setup (cache in session_state)
    if "model" not in st.session_state:
        llm = HuggingFaceEndpoint(
            repo_id="moonshotai/Kimi-K2-Thinking",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            provider="auto",
            add_to_git_credential=True
        )
        st.session_state.model = ChatHuggingFace(llm=llm)
    model = st.session_state.model

    # Session state for conversation
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown("---")
    st.header("Ask questions about the video!")
    user_input = st.text_input("Your question:")

    if st.button("Ask") and user_input:
        formatted_history = "\n".join([
            f"Human: {msg['human']}\nAI: {msg['ai']}" for msg in st.session_state.history
        ])
        context_text = vector_store.similarity_search(user_input, k=3)
        context_text = "\n\n".join([doc.page_content for doc in context_text])
        full_context = formatted_history + "\n" + context_text if formatted_history else context_text
        prompt = augment_fn(full_context, user_input, model)
        answer = model.invoke(prompt)
        st.session_state.history.append({"human": user_input, "ai": answer.content})

    if st.session_state.history:
        st.markdown("## Conversation History")
        for msg in st.session_state.history[::-1]:
            st.markdown(f"**You:** {msg['human']}")
            st.markdown(f"**Bot:** {msg['ai']}")




