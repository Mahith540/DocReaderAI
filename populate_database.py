from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import shutil

CHROMA_PATH = "chroma"
DATA_FOLDERS = ["data_big", "data_text"]  # you can add more folders

def main():
    all_docs = []
    for folder in DATA_FOLDERS:
        all_docs.extend(load_documents(folder))

    if not all_docs:
        print("⚠️ No valid documents found! Please check your data folders.")
        return

    chunks = split_text(all_docs)
    save_to_chroma(chunks)


def load_documents(folder_path):
    """Loads PDFs or text files safely from a folder"""
    docs = []
    print(f"\n📂 Loading documents from: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return docs

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            if filename.lower().endswith(".pdf"):
                print(f"→ Loading PDF: {filename}")
                loader = UnstructuredPDFLoader(file_path)
            elif filename.lower().endswith((".txt", ".md")):
                print(f"→ Loading Text: {filename}")
                loader = TextLoader(file_path)
            else:
                print(f"⚠️ Skipping unsupported file: {filename}")
                continue

            loaded = loader.load()
            if not loaded:
                print(f"⚠️ Skipped empty file: {filename}")
                continue

            docs.extend(loaded)
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

    print(f"✅ Loaded {len(docs)} total documents from {folder_path}.")
    return docs


def split_text(documents):
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks):
    if not chunks:
        print("⚠️ No chunks to save, skipping Chroma creation.")
        return

    print("💾 Saving to Chroma database...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks,
        OllamaEmbeddings(model="mistral"),
        persist_directory=CHROMA_PATH,
    )
    print("✅ Database saved successfully!")


if __name__ == "__main__":
    main()

