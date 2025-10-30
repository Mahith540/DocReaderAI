from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

def query_rag(question: str):
    """Query the local Chroma vector database using Ollama."""
    print(f"🤔 Question: {question}")

    # Load embeddings and Chroma vector DB
    embedding_function = OllamaEmbeddings(model="mistral")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Set up LLM for answering questions
    llm = OllamaLLM(model="mistral")

    # Create a RetrievalQA chain
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # Run query
    result = qa_chain.invoke({"query": question})
    print("\n💡 Answer:", result["result"])
    return result["result"]

if __name__ == "__main__":
    while True:
        q = input("\nAsk me something (or type 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        query_rag(q)

