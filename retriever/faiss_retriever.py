
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        self.index = faiss.read_index("faiss.index")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query, k=4):
        q = self.model.encode([query])
        _, idx = self.index.search(np.array(q), k)
        return idx[0]
