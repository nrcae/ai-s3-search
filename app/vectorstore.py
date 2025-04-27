import faiss

class FAISSVectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []
        self.is_ready = False

    def add(self, vectors, texts):
        self.index.add(vectors.astype("float32"))
        self.text_chunks.extend(texts)
        self.is_ready = True

    def search(self, query_vector, top_k=5):
        if not self.is_ready or self.index.ntotal == 0:
            return ["Index is still being built. Please try again later."]
        D, I = self.index.search(query_vector.astype("float32"), top_k)
        return [self.text_chunks[i] for i in I[0]]