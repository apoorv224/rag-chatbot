# ingest.py
import os
import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

DOCUMENTS_PATH = "docs"
CHROMA_DB_PATH = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "nomic-embed-text"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

def load_txt_files(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                docs.append((f.read(), file))
    return docs

def load_pdf_files(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            with open(os.path.join(folder_path, file), "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                docs.append((text, file))
    return docs

txt_docs = load_txt_files(DOCUMENTS_PATH)
pdf_docs = load_pdf_files(DOCUMENTS_PATH)
all_docs = txt_docs + pdf_docs

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

texts = []
metadatas = []

for text, filename in all_docs:
    chunks = chunk_text(text)
    for chunk in chunks:
        texts.append(chunk)
        metadatas.append({"source": filename})

vectordb.add_texts(texts=texts, metadatas=metadatas)
vectordb.persist()

print(f"✅ Ingested {len(texts)} chunks into ChromaDB at '{CHROMA_DB_PATH}'")