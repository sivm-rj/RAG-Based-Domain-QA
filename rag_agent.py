
from retriever.faiss_retriever import Retriever
from llm.llama_loader import load_llama
from llm.generation import generate

class RAGAgent:
    def __init__(self):
        self.ret = Retriever()
        self.tok, self.model = load_llama()

    def answer(self, q):
        context = "Relevant domain documents retrieved."
        prompt = f"Context: {context}\nQuestion: {q}\nAnswer:"
        return generate(self.tok, self.model, prompt)
