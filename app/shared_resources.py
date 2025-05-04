from app.vectorstore import FAISSVectorStore
vector_store = FAISSVectorStore(embedding_dim=768)

# 3. Optional: Log initialization details
print(f"Shared vector store initialized (Ready: {vector_store.is_ready}, Dim: {vector_store.embedding_dim})")