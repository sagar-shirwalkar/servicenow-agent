#!/usr/bin/env python3
"""
Local RAG system using ChromaDB with metadata-aware retrieval.
Includes byte-level pre-filtering and real-time progress tracking.
"""

import time
import chromadb
from chromadb.utils import embedding_functions
import ollama

# Initialize embedding function (local, via Ollama)
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

# Persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

def get_collection():
    """Get or create the knowledge collection with proper configuration."""
    return client.get_or_create_collection(
        name="servicenow_knowledge",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"}
    )

def index_data(dataset: dict[str, list[dict]], force: bool = False):
    """
    Index categorized chunks into ChromaDB with rich metadata.
    Includes byte-level pre-filtering to prevent Ollama 400 errors.
    """
    collection = get_collection()
    
    count = collection.count()
    if count > 0 and not force:
        print(f"    Collection already has {count} docs. Use force=True to re-index.")
        return
    
    if force and count > 0:
        client.delete_collection("servicenow_knowledge")
        collection = get_collection()
        print(f"    Cleared previous index ({count} docs)")
    
    # Nomic-embed-text has an 8192 token limit. 
    # 1 token ~= 4 bytes. 20000 bytes is a very safe ~5000 token limit.
    MAX_EMBED_BYTES = 20000 
    BATCH_SIZE = 30 
    
    for category, chunks in dataset.items():
        if not chunks:
            continue
            
        print(f"\n   Preparing {len(chunks)} {category} chunks...")
        
        # 1. PRE-FILTERING (Byte-level check to prevent 400 errors)
        valid_docs, valid_ids, valid_metas = [], [], []
        skipped = 0
        
        for i, c in enumerate(chunks):
            doc = c["text"]
            doc_bytes = len(doc.encode('utf-8'))
            
            if doc_bytes > MAX_EMBED_BYTES:
                # Hard truncate at byte level, then decode safely
                doc = doc.encode('utf-8')[:MAX_EMBED_BYTES].decode('utf-8', 'ignore')
            
            if not doc.strip():
                skipped += 1
                continue
                
            valid_docs.append(doc)
            valid_ids.append(f"{category}_{i}_{hash(doc[:50])}")
            valid_metas.append({
                "category": c["category"], 
                "topic": c["topic"],
                "source_file": c["source_file"], 
                "heading": c["heading"],
                "is_code": c.get("is_code", False)
            })
            
        if skipped > 0:
            print(f"         Skipped {skipped} empty chunks")
            
        print(f"          {len(valid_docs)} valid chunks ready for embedding.")
        
        # 2. BATCHING WITH PROGRESS TRACKING
        start_time = time.time()
        indexed_count = 0
        total_valid = len(valid_docs)
        
        for i in range(0, total_valid, BATCH_SIZE):
            batch_docs = valid_docs[i:i+BATCH_SIZE]
            batch_ids = valid_ids[i:i+BATCH_SIZE]
            batch_metas = valid_metas[i:i+BATCH_SIZE]
            
            try:
                collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metas
                )
                indexed_count += len(batch_docs)
                
                # --- PROGRESS TRACKING LOGIC ---
                elapsed = time.time() - start_time
                rate = indexed_count / elapsed if elapsed > 0 else 0
                remaining = total_valid - indexed_count
                eta_mins = (remaining / rate) / 60 if rate > 0 else 0
                
                # \r returns the cursor to the start of the line to overwrite it
                print(f"       {indexed_count}/{total_valid} ({rate:.1f} chunks/s) | ETA: {eta_mins:.1f} min", end='\r')
                
            except Exception as e:
                print(f"\nBatch failed: {e}. Retrying individually...")
                for d, id_, m in zip(batch_docs, batch_ids, batch_metas):
                    try:
                        collection.add(documents=[d], ids=[id_], metadatas=[m])
                        indexed_count += 1
                    except:
                        pass # Drop irrecoverable chunks silently
                        
        print(f"\n         Finished {category}: {collection.count()} total docs in DB.")

def retrieve_context(
    query: str, 
    n_results: int = 3,
    category_filter: str | None = None,
    topic_filter: str | None = None
) -> str:
    """
    Retrieve relevant chunks with optional metadata filtering.
    Includes safety truncation to protect the LLM context window.
    """
    collection = get_collection()
    
    where_clause = {}
    if category_filter:
        where_clause["category"] = category_filter
    if topic_filter:
        where_clause["topic"] = topic_filter
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause if where_clause else None,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results['documents'][0]:
        return ""
    
    formatted = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        header = f"[{meta['topic'].title()} | {meta['heading']}]"
        if meta.get('is_code'):
            header += " (Code Example)"
        formatted.append(f"{header}\n{doc}\n")
    
    combined_text = "\n---\n".join(formatted)
    
    # Safety truncation for the LLM context window (approx 4000 tokens)
    MAX_CONTEXT_CHARS = 15000 
    if len(combined_text) > MAX_CONTEXT_CHARS:
        combined_text = combined_text[:MAX_CONTEXT_CHARS] + "\n\n[... Context truncated ...]"
        
    return combined_text

def retrieve_cross_topic(
    query: str,
    topics: list[str],
    n_per_topic: int = 2
) -> str:
    """
    Specialized retrieval for cross-topic workflow synthesis.
    """
    all_chunks = []
    
    for topic in topics:
        chunks = retrieve_context(
            query,
            n_results=n_per_topic,
            topic_filter=topic
        )
        if chunks:
            all_chunks.append(f"\n## {topic.title()} Context:\n{chunks}")
    
    return "\n".join(all_chunks) if all_chunks else retrieve_context(query, n_results=5)
