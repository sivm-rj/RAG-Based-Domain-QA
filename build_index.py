
import os, faiss
from embeddings.embedding_utils import load_model, embed

docs = []
for f in os.listdir("data/raw/documents"):
    with open(f"data/raw/documents/{f}") as file:
        docs.append(file.read())

model = load_model("sentence-transformers/all-MiniLM-L6-v2")
vecs = embed(model, docs)

index = faiss.IndexFlatL2(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, "faiss.index")
print("FAISS index built")
