# RAG Chatbot 🤖:

1. A Retrieval-Augmented Generation (RAG) chatbot built as a personal GenAI project.
2. It allows users to ask questions over custom documents using embeddings and vector search.


## Features:

- Document ingestion and chunking
- Vector embeddings
- Semantic search using ChromaDB
- LLM-powered responses


## Project Structure:

```text
RAG_Chatbot_GenAI/
├── src/
│   ├── chatbot.py        # Handles user queries + RAG pipeline
│   └── ingest.py         # Loads & embeds documents
├── data/
│   └── sample_docs/      # Source documents for RAG
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup:

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Create a .env file:
   OPENAI\_API\_KEY=your\_api\_key\_here


## Run:

1. python src/ingest.py
2. python src/chatbot.py


## Learning Goals:

- Understand RAG pipelines
- Practice vector databases
- Build real GenAI applications
