
from fastapi import FastAPI
from agents.rag_agent import RAGAgent

app = FastAPI()
agent = RAGAgent()

@app.get("/qa")
def qa(query: str):
    return {"answer": agent.answer(query)}
