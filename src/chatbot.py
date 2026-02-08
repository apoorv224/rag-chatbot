from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama


CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama2" 
TOP_K = 3


embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectordb = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)


memory = []


def retrieve_chunks(query, vectordb, top_k=TOP_K):
    docs = vectordb.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]


def rag_query(query, vectordb, memory):
    context_chunks = retrieve_chunks(query, vectordb)
    context = "\n\n".join(context_chunks)

    history = ""
    for q, a in memory[-3:]:
        history += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
You are a helpful assistant.
Answer ONLY from the given context.
If the answer is not in the context, say "Not mentioned in the document."

Conversation history:
{history}

Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"].strip()
    memory.append((query, answer))
    return answer


print("RAG Chatbot Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    answer = rag_query(query, vectordb, memory)
    print(f"Bot: {answer}\n")