# src/ingest.py
import os
import sys
import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

DOCUMENTS_PATH = os.path.join(PROJECT_ROOT, "docs")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "nomic-embed-text"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectordb = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

def load_txt_files(folder_path):
    docs = []
    if not os.path.exists(folder_path):
        return docs

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append((f.read(), file))
    return docs


def load_pdf_files(folder_path):
    docs = []
    if not os.path.exists(folder_path):
        return docs

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                docs.append((text, file))
    return docs

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def ingest_documents():
    txt_docs = load_txt_files(DOCUMENTS_PATH)
    pdf_docs = load_pdf_files(DOCUMENTS_PATH)

    all_docs = txt_docs + pdf_docs

    if not all_docs:
        print("⚠️ No documents found in docs/ folder")
        return

    texts = []
    metadatas = []

    for text, filename in all_docs:
        chunks = chunk_text(text)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": filename})

    vectordb.add_texts(texts=texts, metadatas=metadatas)
    vectordb.persist()

    print(f"✅ Ingested {len(texts)} chunks into ChromaDB")
    print(f"📂 ChromaDB location: {CHROMA_DB_PATH}")

if __name__ == "__main__":
    ingest_documents()