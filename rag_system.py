import chromadb
from chromadb.utils import embedding_functions
import ollama

#
# We will use ChromaDB in persistent local mode. 
# It requires zero external servers and runs entirely in-process.
# 

# Use Ollama's nomic-embed-text for embeddings
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="servicenow_knowledge",
    embedding_function=ollama_ef
)

def index_data(dataset):
    """Indexes the categorized data into ChromaDB."""
    for category, chunks in dataset.items():
        ids = [f"{category}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": category} for _ in chunks]
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)

def retrieve_context(query: str, n_results=3):
    """Retrieves the most relevant chunks for a given query."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return "\n\n---\n\n".join(results['documents'][0])
