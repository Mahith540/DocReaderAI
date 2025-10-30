from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    # Use a local Ollama embedding model
    return OllamaEmbeddings(model="mxbai-embed-large")

