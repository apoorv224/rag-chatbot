RAG Chatbot 🤖:
A Retrieval-Augmented Generation (RAG) chatbot built as a personal GenAI project.
It allows users to ask questions over custom documents using embeddings and vector search.

Features:
\- Document ingestion and chunking
\- Vector embeddings
\- Semantic search using ChromaDB
\- LLM-powered responses

Project Structure:

RAG\_Chatbot\_GenAI/

├── src/

│   ├── chatbot.py

│   └── ingest.py

├── data/

│   └── sample\_docs/

├── requirements.txt

├── .gitignore

└── README.md

Setup:

1\. Clone the repository
2\. Install dependencies:
   pip install -r requirements.txt
3\. Create a .env file:
   OPENAI\_API\_KEY=your\_api\_key\_here

Run:

python src/ingest.py
python src/chatbot.py

Learning Goals:
\- Understand RAG pipelines
\- Practice vector databases
\- Build real GenAI applications
