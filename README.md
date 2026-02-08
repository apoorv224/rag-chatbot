\# RAG Chatbot 🤖



A Retrieval-Augmented Generation (RAG) chatbot built as a personal GenAI project.

It allows users to ask questions over custom documents using embeddings and vector search.



\## Features

\- Document ingestion and chunking

\- Vector embeddings

\- Semantic search using ChromaDB

\- LLM-powered responses



\## Project Structure

RAG\_Chatbot\_GenAI/

├── chatbot.py        # Chat interface

├── ingest.py         # Document ingestion

├── requirements.txt  # Dependencies

├── .gitignore

└── README.md



\## Setup

1\. Clone the repository

2\. Install dependencies:

&nbsp;  pip install -r requirements.txt



3\. Create a .env file:

&nbsp;  OPENAI\_API\_KEY=your\_api\_key\_here



\## Run

python ingest.py

python chatbot.py



\## Learning Goals

\- Understand RAG pipelines

\- Practice vector databases

\- Build real GenAI applications

