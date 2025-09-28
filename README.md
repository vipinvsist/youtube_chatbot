# YouTube Chatbot

This project is a YouTube chatbot that fetches video transcripts and allows users to ask questions based on the content of those transcripts. It utilizes various libraries to process the text, generate embeddings, and provide relevant answers.

## Project Structure

```
l_youtube_chatbot
├── a_data_ingestion.py      # Fetches YouTube video transcripts
├── b_text_splitter.py       # Splits text into smaller chunks
├── c_embeddings.py           # Generates vector embeddings for text chunks
├── d_similarity_search.py    # Retrieves similar documents from the vector store
├── e_augmentation.py         # Augments questions based on context
├── chain.py                  # Orchestrates the entire process
└── generation.py             # Generates answers to specific questions
├── requirements.txt          # Lists project dependencies
├── .env
└── README.md                     # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/vipinvsist/youtube_chatbot.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Fill `.env` with the necessary values.

## Usage

1. To fetch a transcript and ask a question, run:
   ```
   python chain.py
   ```

2. You can also use the `generation.py` script to generate answers based on specific questions.

## Dependencies

This project requires the following Python packages:
- `youtube-transcript-api`
- `langchain`
- `python-dotenv`
