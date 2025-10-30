from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
import shutil

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data_text"

def main():
    generate_data_store()

def generate_data_store():
    print(f"📂 Loading documents from: {DATA_PATH}")
    documents = load_documents()
    print(f"✅ Loaded {len(documents)} text documents.")
    
    print("✂️ Splitting documents into chunks...")
    chunks = split_text(documents)
    print(f"✅ Split into {len(chunks)} chunks.")
    
    print("💾 Saving to Chroma database...")
    save_to_chroma(chunks)
    print("✅ Chroma database ready! 🚀")

def load_documents():
    """Load all .txt files from the data_text folder."""
    docs = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            print(f"→ Loaded TXT: {filename}")
            loader = TextLoader(os.path.join(DATA_PATH, filename))
            docs.extend(loader.load())
    return docs

def split_text(documents):
    """Split text into chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

def save_to_chroma(chunks):
    """Save text chunks into a Chroma vector database using Ollama embeddings."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create and auto-persist database
    Chroma.from_documents(
        chunks,
        OllamaEmbeddings(model="mistral"),
        persist_directory=CHROMA_PATH
    )

if __name__ == "__main__":
    main()

