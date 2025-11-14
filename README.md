# YouTube Chatbot

This project is a YouTube chatbot that fetches video transcripts and allows users to ask questions based on the content of those transcripts. It utilizes modern LLMs (HuggingFace) to process the text, generate embeddings, and provide relevant answers. The project includes both a command-line and a Streamlit web interface.

## Features
- Fetches YouTube video transcripts by video ID
- Splits transcripts into manageable text chunks
- Generates vector embeddings for semantic search
- Retrieves relevant context for user questions
- Augments and answers questions using LLMs (HuggingFace)
- Remembers conversation history during a session
- Interactive Streamlit UI with video display and chat

## Project Structure
```
youtube_chatbot/
├── a_data_ingestion.py      # Fetches YouTube video transcripts
├── b_text_splitter.py       # Splits text into smaller chunks
├── c_embeddings.py          # Generates vector embeddings for text chunks
├── d_similarity_search.py   # Retrieves similar documents from the vector store
├── e_augmentation.py        # Augments questions based on context
├── chain.py                 # Command-line chatbot with memory
├── generation.py            # Simple Q&A script
├── streamlit_app.py         # Streamlit web app for interactive chat
├── requirements.txt         # Project dependencies
├── .env                     # Environment variables (API keys, etc.)
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vipinvsist/youtube_chatbot.git
   cd youtube_chatbot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file and add any required API keys (e.g., for HuggingFace).

## Usage

### Command-Line Chatbot
Run the chatbot and follow the prompts:
```bash
python chain.py
```
- Enter the YouTube video ID when prompted.
- Ask questions about the video transcript.
- Type `exit` or `quit` to end the session.

### Streamlit Web App
Launch the interactive web UI:
```bash
streamlit run streamlit_app.py
```
- Enter the YouTube video ID in the sidebar.
- The video and chat interface will appear.
- Ask questions and view the conversation history.

### Simple Q&A Script
For a single question/answer:
```bash
python generation.py
```

## Dependencies

This project requires the following Python packages:
- `youtube-transcript-api`
- `langchain`
- `langchain_huggingface`
- `streamlit`
- `python-dotenv`
- `faiss-cpu` (or `faiss-gpu`)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Notes
- The chatbot uses only HuggingFace models (no OpenAI key required).
- All modules are reusable and not tied to any specific video.
- For best results, use English-language YouTube videos with available transcripts.

---

Feel free to contribute or open issues for improvements!
