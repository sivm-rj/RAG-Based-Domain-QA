
from agents.rag_agent import RAGAgent

agent = RAGAgent()
while True:
    q = input("Ask: ")
    if q == "exit":
        break
    print(agent.answer(q))
