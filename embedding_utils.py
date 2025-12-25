
from sentence_transformers import SentenceTransformer

def load_model(name):
    return SentenceTransformer(name)

def embed(model, texts):
    return model.encode(texts, convert_to_numpy=True)
